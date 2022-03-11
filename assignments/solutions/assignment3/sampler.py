import math
from typing import List

import torch
from ray_utils import RayBundle, get_device
from pytorch3d.renderer.cameras import CamerasBase
from render_functions import render_points

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
        print('+++')
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = get_device())

        # TODO (1.4): Sample points from z values
        n_rays = ray_bundle.origins.shape[0]
        sample_points = torch.zeros((n_rays, self.n_pts_per_ray, 3), device = get_device())
        for i in range(n_rays):
            # for each ray, sample n points
            scale = z_vals / ray_bundle.directions[i][-1] # m
            sample_points[i] = scale.unsqueeze(-1) * ray_bundle.directions[i].reshape(1,3)
            # print(sample_points[i].shape)

        # visualize
        # print('createing ray visualizaiton')
        # print(sample_points.shape)
        # render_points('1.4.png', sample_points[25105].unsqueeze(0), image_size=256, color=[0.7, 0.7, 1])
        # print('done')
        
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}