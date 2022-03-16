import torch

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
        # print('density:', rays_density)
        # print('deltas:', deltas)

        # TODO (1.5): Compute transmittance using the equation described in the README
        n_rays, n_sample_per_ray = rays_density.shape[0], rays_density.shape[1]
        T = torch.zeros(n_rays, n_sample_per_ray).to(get_device())

        T[:,0] = 1.0 # first segment = 1
        for j in range(1, n_sample_per_ray):
            T[:,j] = T[:, j-1] * torch.exp(-rays_density[:, j, 0] * deltas[:, j-1, 0])

        # TODO (1.5): Compute weight used for rendering from transmittance and density
        weights = T.reshape(-1,1) * (1.-torch.exp(-rays_density.reshape(-1,1)*deltas.reshape(-1,1)))
        weights = weights.reshape(n_rays, n_sample_per_ray)

        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # # TODO (1.5): Aggregate (weighted sum of) features using weights
        n_rays, n_samples = weights.shape[0], weights.shape[1]
        # rays_feature = rays_feature.reshape(n_rays, n_samples, 3)

        feature = weights.reshape(-1,1) * rays_feature
        feature = feature.reshape(n_rays, n_samples, 3)
        feature = torch.sum(feature, dim = 1)

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
            # print(density)
            # print(feature)

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
            feature = self._aggregate(weights, feature)

            # TODO (1.5): Render depth map
            depth = torch.ones((n_pts, 1))

            # Return
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
