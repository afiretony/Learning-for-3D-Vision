import argparse
import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import random
import math

from starter.utils import get_device, get_mesh_renderer, get_points_renderer


def bullet_time(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/",
):
    if device is None:
        device = get_device()
    verts, faces, aux = pytorch3d.io.load_obj("data/cow.obj")
    
    # print(faces.verts_idx)
    faces = faces.verts_idx
    # print(faces.verts_idx.shape)

    num_triangle = faces.shape[0]
    areas = torch.zeros((num_triangle))
    for i in range(num_triangle):
        v1 = verts[faces[i][0]][:]
        v2 = verts[faces[i][1]][:]
        v3 = verts[faces[i][2]][:]
        areas[i] = abs(0.5 * torch.inner(v1-v2, v1-v3))
    
    weight = areas / sum(areas)

    num_samples = 100 # number of point cloud
    sampled_faceidx = []
    for i in range(num_samples):
      rnd = random.uniform(0, 1)
      for j, w in enumerate(weight):
          if w<0:
              raise ValueError("Negative weight encountered.")
          rnd -= w
          if rnd < 0:
              sampled_faceidx.append(j)
              break

    print("number of samples generated is:", num_samples)
    if num_samples != len(sampled_faceidx): raise ValueError("WTF?")

    points = torch.zeros((num_samples,3))
    for i in range(num_samples):
      idx = sampled_faceidx[i]
      p1, p2, p3 = verts[faces[idx][0]][:], verts[faces[idx][1]][:], verts[faces[idx][2]][:]
      alpha = random.uniform(0, 1)
      alpha2 = random.uniform(0, 1)
      alpha1 = 1- math.sqrt(alpha)
      v = alpha1 * p1 + (1-alpha1)*alpha2*p2 + (1-alpha1)*(1-alpha2)*p3
      points[i] = v

    color = (points - points.min()) / (points.max() - points.min())
    cow_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features = [color]
    ).to(device)

    renders = []
    angles = np.linspace(0,360,num_frames)
    for i, angle in enumerate(tqdm(angles)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=5.0, elev=2, azim=angle)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(cow_point_cloud, cameras=cameras)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="output/extra100.gif")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    bullet_time(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
