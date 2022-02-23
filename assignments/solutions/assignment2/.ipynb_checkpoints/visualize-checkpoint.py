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
    color=[253/255, 185/255,200/255]
    vertices, faces = mesh_src.verts_list()[0], mesh_src.faces_list()[0]

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures.to(device) * torch.tensor(color).to(device)  # (1, N_v, 3)
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
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=1.5, elev=0.5, azim=90, at=((0, 0, 0), ))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
    rend = renderer(render_point_cloud, cameras=cameras)

    image = (rend*255).detach().cpu().numpy().astype(np.uint8).reshape((256,256,3))
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(outpput_path,image)
    plt.axis("off");
    return


def visualize_voxel_likelyhood(voxels, output_path):
    image_size = 512
    device = get_device()
    levels = 10
    voxels = torch.sigmoid(voxels)
    print(voxels)

    steps = np.linspace(0.4, 0.99, num=levels)
    prev_step = 0.0
    allverts = torch.empty((1,1,3)).to(device)
    allfaces = torch.empty((1,1,3)).to(device)
    alltexts = torch.empty((1,1,3)).to(device)

    meshes = []
    for step in steps:
        sub_voxels = voxels.clone()
        sub_voxels[sub_voxels>step]=0.1
        sub_voxels[sub_voxels<prev_step]=0.1
        prev_step = step
        
        sub_mesh = cubify(sub_voxels, thresh=0.3).to(device)
        
        vertices, faces = sub_mesh.verts_list()[0], sub_mesh.faces_list()[0]
        if vertices.shape[0]>10:

            vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
            # allverts = torch.cat([allverts, vertices], dim = 1)

            faces = faces.unsqueeze(0)+torch.max(allfaces)  # (N_f, 3) -> (1, N_f, 3)
            # allfaces = torch.cat([allfaces, faces], dim = 1)

            color = [step, 0.0, 1.0-step]
            textures = torch.ones_like(vertices).to(device)  # (1, N_v, 3)
            textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)
            # alltexts = torch.cat([alltexts, textures], dim = 1)
            mesh = pytorch3d.structures.Meshes(
                verts=vertices.to(torch.float32),
                faces=faces.to(torch.float32),
                textures=pytorch3d.renderer.TexturesVertex(textures.to(torch.float32)),
            )
            meshes.append(mesh)
            
    # join mesh
    allmesh = pytorch3d.structures.join_meshes_as_scene(meshes, include_textures=True)
        
            

    # color=[253/255, 0 , ]
#     vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]
    
#     vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
#     faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
#     textures = torch.ones_like(vertices)  # (1, N_v, 3)
#     textures = textures * torch.tensor(color).to(device)  # (1, N_v, 3)

    # print(allverts.shape)
    # print(allfaces)
    # print(alltexts.shape)
    
    # mesh = pytorch3d.structures.Meshes(
    #     verts=allverts,
    #     faces=allfaces,
    #     textures=pytorch3d.renderer.TexturesVertex(alltexts),
    # )
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 3.0]], device=device)

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=2.5, elev=1, azim=90, at=((0, 0, 0), ))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
    rend = renderer(allmesh, cameras=cameras, lights=lights)
    image = (rend*255).detach().cpu().numpy().astype(np.uint8).squeeze(0)
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(output_path,image)
    plt.axis("off");
    