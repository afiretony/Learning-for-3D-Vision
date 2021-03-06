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

from starter.utils import get_device, get_mesh_renderer


def bullet_time(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/creative.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 10.0]], device=device)

    renders = []
    angles = np.linspace(0,360, num_frames)
    for i, angle in enumerate(tqdm(angles)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=1000.0, elev=15.0, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T, R=R, device=device)
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
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--output_file", type=str, default="output/helico.gif")
    parser.add_argument("--image_size", type=int, default=720)
    args = parser.parse_args()
    bullet_time(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
