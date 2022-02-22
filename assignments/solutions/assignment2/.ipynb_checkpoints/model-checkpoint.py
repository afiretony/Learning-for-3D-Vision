from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            in_channels = 64
            channels = [32, 16, 8]
            modules = []
            for mid_channels in channels:
                modules += [
                    nn.ConvTranspose3d(in_channels=in_channels, out_channels=mid_channels, 
                                       kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(mid_channels),
                    nn.ReLU()
                ]
                in_channels = mid_channels

            # output layer
            out_channels = 1
            modules += [
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
            ]
            self.decoder = torch.nn.Sequential(*modules)
        
        elif args.type == "point":
            self.n_point = args.n_points
            in_channels = 512
            channels = [256, 128, 64, 32, 16]
            modules = []
            for mid_channels in channels:
                modules += [
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, 
                                       kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU()
                ]
                in_channels = mid_channels

            # output layer
            out_channels = 3
            modules += [
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Flatten(),
                nn.Linear(12288, self.n_point*3)
            ]

            self.decoder = torch.nn.Sequential(*modules)
            
        elif args.type == "mesh":
            # try different mesh initializations
            mesh_pred = ico_sphere(4,'cuda')
            
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =             

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        # call decoder
        batch_size = encoded_feat.shape[0]
        
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat.reshape((batch_size, 64, 2, 2, 2)))
            return voxels_pred

        elif args.type == "point":
            # TODO:
            
            pointclouds_pred = self.decoder(encoded_feat.reshape((batch_size, 512, 1, 1)))
            return pointclouds_pred.reshape((batch_size, args.n_points, 3))

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

