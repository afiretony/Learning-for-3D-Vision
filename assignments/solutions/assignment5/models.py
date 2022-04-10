import torch
import torch.nn as nn
import torch.nn.functional as F

class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1) # Lin=Lout=N
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.f1 = nn.Linear(1024, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # shared weights MLP
        points = torch.transpose(points, 1, 2) # B, 3, N
        out = self.relu(self.bn1(self.conv1(points))) # B, 64, N
        out = self.relu(self.bn2(self.conv2(out))) # B, 128, N
        out = self.relu(self.bn3(self.conv3(out))) # B, 1024, N
        
        # max pool
        out, _ = torch.max(out, dim=-1) # global feature, B x 1024

        # MLP
        # out = self.relu(self.f1(out))
        # out = self.relu(self.f2(out))
        out = self.f1(out)
        out = self.f2(out)
        out = nn.functional.softmax(self.f3(out), -1) # B x k
        return out



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1) # Lin=Lout=N
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_seg_classes, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
         # shared weights MLP
        points = torch.transpose(points, 1, 2) # B, 3, N
        local_embedding = self.relu(self.bn1(self.conv1(points))) # B, 64, N

        out = self.relu(self.bn2(self.conv2(local_embedding))) # B, 128, N
        out = self.relu(self.bn3(self.conv3(out))) # B, 1024, N
        
        # max pool
        global_embedding, _ = torch.max(out, dim=-1) # global feature, B x 1024

        N = local_embedding.shape[-1] # num of points per object
        B = global_embedding.shape[0]
        # print('--------')
        combined = torch.cat(
            (torch.transpose(local_embedding, 1, 2), # B x N x 64
            global_embedding.repeat(1, N).view(B, N, 1024)), # B x N x 1024
            dim=-1
        ).transpose(1, 2) # B x 1088 x N

        out = self.relu(self.bn4(self.conv4(combined))) # B x 512 x N
        out = self.relu(self.bn5(self.conv5(out))) # B x 256 x N
        out = self.relu(self.bn6(self.conv6(out))) # B x 128 x N
        out = nn.functional.softmax(self.conv7(out), dim=1) # B x m x N
        return out.transpose(1, 2) 
        






        
        
        



