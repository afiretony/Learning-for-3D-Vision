"""
Usage:
    python -m starter.dolly_zoom --num_frames 10
"""

import argparse

import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer, load_tetrahedron_mesh


def bullet_time(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/",
):
    if device is None:
        device = get_device()

    # Get the vertices, faces, and textures.
    color=[0.7, 0.7, 1]
    vertices, faces = load_tetrahedron_mesh()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    renders = []
    angles = np.linspace(0,360, num_frames)
    for i, angle in enumerate(tqdm(angles)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=6.0, elev=0.0, azim=angle, at=((0, 1, 0), ))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T, device=device)
        cameras.get_full_projection_transform(R=R)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"angle: {angles[i]:.2f}", fill=(0, 0, 255))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, fps=(num_frames / duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output_file", type=str, default="output/q2.1.gif")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    bullet_time(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
