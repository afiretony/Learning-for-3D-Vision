import argparse
import pickle
from tqdm.auto import tqdm
import imageio
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_plant(
    image_size=512,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    def common_render(output_file):
        renders = []
        angles = np.linspace(0,180,num_frames)
        for i, angle in enumerate(tqdm(angles)):
            R, T = pytorch3d.renderer.look_at_view_transform(dist=6.0, elev=10, azim=angle)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
            rend = renderer(point_cloud, cameras=cameras)
            rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            renders.append(rend)

        images = []
        for i, r in enumerate(renders):
            image = Image.fromarray((r * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), f"angle: {angles[i]:.2f}", fill=(0, 0, 255))
            images.append(np.array(image))
        imageio.mimsave(output_file, images, fps=(num_frames / duration))

    # parameters of gif    
    duration = 5
    num_frames = 2

    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    # rander the first image
    data = load_rgbd_data()
    rgb = torch.from_numpy(data['rgb1']).to(device)
    depth = torch.from_numpy(data['depth1']).to(device)
    mask = torch.from_numpy(data['mask1']).to(device)

    camera = data['cameras1']
    verts1, rgba1 = unproject_depth_image(rgb, mask, depth, camera)
    verts1 = verts1.unsqueeze(0)
    rgb = rgba1[:,:3].unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts1, features=rgb)
    
    output_file = 'output/q5_1_1.gif'
    common_render(output_file)

    # render second image
    rgb = torch.from_numpy(data['rgb2']).to(device)
    depth = torch.from_numpy(data['depth2']).to(device)
    mask = torch.from_numpy(data['mask2']).to(device)
    camera = data['cameras2']
    verts2, rgba2 = unproject_depth_image(rgb, mask, depth, camera)
    verts2 = verts2.unsqueeze(0)
    rgb = rgba2[:,:3].unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts2, features=rgb)

    output_file = 'output/q5_1_2.gif'
    common_render(output_file)

    verts_cat = torch.cat((verts1, verts2),1)
    rgb_cat = torch.cat((rgba1[:,:3], rgba2[:,:3]),0).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts_cat, features=rgb_cat)

    output_file = 'output/q5_1_3.gif'
    common_render(output_file)

if __name__ == "__main__":
    render_plant()
    # plt.imsave('output/plant1.jpg', image)
