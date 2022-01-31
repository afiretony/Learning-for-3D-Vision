"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import math

from starter.utils import get_device, get_mesh_renderer


# def render_textured_cow(
#     cow_path="data/cow.obj",
#     image_size=256,
#     R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#     T_relative=[0, 0, 0],
#     device=None,
# ):
#     if device is None:
#         device = get_device()
#     meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
#     rotate_angle = 0. / 180. * math.pi
#     # R_relative=[math.cos(rotate_angle), 0, -math.sin(rotate_angle)], [0, 1, 0], [math.sin(rotate_angle), 0, math.cos(rotate_angle)]
#     T_relative=[0., 0, 1]
#     print(R_relative)
#     R_relative = torch.tensor(R_relative).float()
#     T_relative = torch.tensor(T_relative).float()
#     R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
#     T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
#     T= T_relative
#     renderer = get_mesh_renderer(image_size=256)
#     cameras = pytorch3d.renderer.FoVPerspectiveCameras(
#         R=R.unsqueeze(0), T=T.unsqueeze(0), device=device, fov=60.
#     )
#     lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
#     rend = renderer(meshes, cameras=cameras, lights=lights)
#     return rend[0, ..., :3].cpu().numpy()
def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    rotate_angle = -90. / 180. * math.pi
    R_relative=[math.cos(rotate_angle), 0, -math.sin(rotate_angle)], [0, 1, 0], [math.sin(rotate_angle), 0, math.cos(rotate_angle)]
    T_relative=[-3., 0, 3]
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    # T = torch.tensor([0.0, 0, 3]) + T_relative
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="output/cam4.jpg")
    args = parser.parse_args()
    render_textured_cow(cow_path=args.cow_path, image_size=args.image_size)
    plt.imsave(args.output_path, render_textured_cow())
