import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import knn_points

# define losses
def voxel_loss(voxel_src,voxel_tgt):
  N = voxel_tgt.shape[1] * voxel_tgt.shape[2]* voxel_tgt.shape[3]
  loss = voxel_tgt * torch.log(voxel_src) + (1.0-voxel_tgt)*torch.log(1.0-voxel_src)
  # implement some loss for binary voxel grids
  prob_loss = -torch.sum(loss, (1,2,3)) / N
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