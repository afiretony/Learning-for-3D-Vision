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

from starter.utils import get_device, get_mesh_renderer, get_points_renderer

def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    # parameters of the gif
    num_frames = 10
    duration = 10
    output_file = 'output/q5_3.gif'

    if device is None:
        device = get_device()
    min_value = -5.
    max_value = 5.
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    R = 2
    r = 0.5
    voxels = (torch.sqrt(X**2+Y**2)-R**2)**2+Z**2-r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)

    renders = []
    angles = np.linspace(0,360,num_frames)
    for i, angle in enumerate(tqdm(angles)):
        renderer = get_mesh_renderer(image_size=image_size, device=device)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=0, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.2f}", fill=(0, 0, 255))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, fps=(num_frames / duration))

if __name__ == "__main__":
    
    render_torus_mesh()