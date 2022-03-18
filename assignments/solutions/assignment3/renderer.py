import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase

from ray_utils import get_device
# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):

        # TODO (1.5): Compute transmittance using the equation described in the README
        n_rays, n_sample_per_ray = deltas.shape[0], deltas.shape[1]

        multiplier = torch.exp(-(deltas * rays_density))
        
        T = torch.ones(n_rays, n_sample_per_ray,1).to(get_device())
        
        for i in range(1, n_sample_per_ray):
            T[:,i,:] = T[:,i-1,:].clone() * multiplier[:,i-1,:] # clone avoids inplace error

        # TODO (1.5): Compute weight used for rendering from transmittance and density
        weights = T * (1-torch.exp(-rays_density*deltas))
        
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor, 
        rays_feature: torch.Tensor
    ):

        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]
        
        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)

            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']
            # print('input feature size:', feature.shape)
            # print('input density size:', density.shape)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]

            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature.view(-1, n_pts, 3))

            # TODO (1.5): Render depth map
            # density: [n x m x 1]
            density = density.view(-1, n_pts, 1)
            depth = torch.zeros((density.shape[0], 1)).to(get_device())
            for j in range(50):
                # find where density is larger than thershold and dpeth not updated
                idx = torch.logical_and(density[:,j,0] > 0.1, depth[:,0] < 0.01 )
                depth[idx] = 1.0 - j / 50
            
            depth = depth / torch.max(depth)


            # print('output feature size:', feature.shape)
            # print('output weight size:', weights.shape)

            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
