import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from utils import get_device, get_mesh_renderer, get_points_renderer
import matplotlib.pyplot as plt
import mcubes

from pytorch3d.ops import cubify

def visualize_mesh(mesh_src, output_path):
    image_size = 512
    device = get_device()
    color=[1.0, 0.,0.]
    vertices, faces = mesh_src.verts_list()[0], mesh_src.faces_list()[0]

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    textures.to(device)
    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 3.0]], device=device)

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=1.5, elev=0.5, azim=90, at=((0, 0, 0), ))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    image = (rend*255).detach().cpu().numpy().astype(np.uint8).squeeze(0)
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(output_path,image)
    plt.axis("off");

def visualize_voxel(voxels, output_path):
    image_size = 512
    device = get_device()
    mesh = cubify(voxels, thresh=0.8).to(device)
    color=[253/255, 185/255,200/255]
    vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 3.0]], device=device)

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=2.5, elev=1, azim=90, at=((0, 0, 0), ))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    image = (rend*255).detach().cpu().numpy().astype(np.uint8).squeeze(0)
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(output_path,image)
    plt.axis("off");
    
def visualize_pcd(point_cloud_src, outpput_path):
    '''
    visualize point cloud
    input: point_cloud_src: torch of size [1, n_points, 3]
    '''
    
    device = get_device()
    # point_cloud_src = point_cloud_src.squeeze(0)
    points = point_cloud_src[0]
    color = (points - points.min()) / (points.max() - points.min())

    render_point_cloud = pytorch3d.structures.Pointclouds(
      points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=256, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 1.5]], device=device)
    rend = renderer(render_point_cloud, cameras=cameras)

    image = (rend*255).detach().cpu().numpy().astype(np.uint8).reshape((256,256,3))
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(outpput_path,image)
    plt.axis("off");
    return

