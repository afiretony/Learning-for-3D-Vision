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
        pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        pass



