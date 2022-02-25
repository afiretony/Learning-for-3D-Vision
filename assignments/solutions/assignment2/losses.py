import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import knn_points

# define losses
def voxel_loss(voxel_src, voxel_tgt):
    # implement some loss for binary voxel grids
    # batch_size = voxel_src.shape[0]

    voxel_src = torch.sigmoid(voxel_src)
    loss = voxel_tgt * torch.log(voxel_src) + (1.0-voxel_tgt)*torch.log(1.0-voxel_src)
    prob_loss = -torch.sum(loss) / torch.numel(voxel_src)
    
    return prob_loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
    # implement chamfer loss from scratch
    N = point_cloud_src.shape[1]
    dists1, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K=1)
    dists2, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K=1)
    # K=1 because we just take the minimum distance
    dists1 = dists1.reshape((-1, N)) # [batch_size, num_points]
    dists2 = dists2.reshape((-1, N))
    loss_chamfer = torch.sum(dists1,1) + torch.sum(dists2,1)
    # loss_chamfer2,_ = chamfer_distance(point_cloud_src, point_cloud_tgt)
    return loss_chamfer

def smoothness_loss(mesh_src):
    loss_laplacian = mesh_laplacian_smoothing(mesh_src)
    # implement laplacian smoothening loss
    return loss_laplacian