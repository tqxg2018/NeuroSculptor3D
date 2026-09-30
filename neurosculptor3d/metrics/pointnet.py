"""PointNet classifier used for the Frechet Point-cloud Distance (FPD).

Architecture from https://github.com/fxia22/pointnet.pytorch (MIT License). The FPD checkpoint
(`checkpoint_300_0.9205357142857142.pth`, 13 ShapeNet classes) is the one used by the MinD-3D evaluation code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        b = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, device=x.device, dtype=x.dtype).flatten()[None].repeat(b, 1)
        return (x + iden).view(-1, 3, 3)


class PointNetfeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        trans = self.stn(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return torch.max(x, 2, keepdim=True)[0].view(-1, 1024), trans


class PointNetCls(nn.Module):
    def __init__(self, k=13):
        super().__init__()
        self.feat = PointNetfeat()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        """x: (B, 3, N). Returns logits and the 1805-d activation (concat of all FC layers) used by FPD."""
        x1, _ = self.feat(x)
        x2 = F.relu(self.bn1(self.fc1(x1)))
        x3 = F.relu(self.bn2(self.fc2(x2)))
        x4 = self.fc3(x3)
        return x4, torch.cat((x1, x2, x3, x4), dim=1)
