import math
from typing import List

import torch
from ray_utils import RayBundle, get_device
from pytorch3d.renderer.cameras import CamerasBase
from render_functions import render_points
from ray_utils import get_device
# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        n_rays = ray_bundle.origins.shape[0]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = get_device())

        # TODO (1.4): Sample points from z values
        sample_points = torch.zeros((n_rays, self.n_pts_per_ray, 3), device = get_device())

        z_vals = torch.ones((n_rays,1)).to(get_device()) @ z_vals.reshape(1, -1)
        z_vals = z_vals.reshape(n_rays, self.n_pts_per_ray, 1)
        
        for i in range(self.n_pts_per_ray):
            sample_points[:,i,:] = ray_bundle.directions

        sample_points = z_vals * sample_points
        for i in range(self.n_pts_per_ray):
            sample_points[:,i,:] = sample_points[:,i,:] + ray_bundle.origins

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        ) 


sampler_dict = {
    'stratified': StratifiedRaysampler
}