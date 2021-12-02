import torch
import torch.nn as nn
import torch.nn.functional as F


class Hierarchical:
    def __init__(self, depth):
        self.type = "hierarchical"
        self.depth = depth
        self.controller = None
        self.submodules = []


class PrimBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "primitive"
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=4, stride=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(4*14*14, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.softmax(out, dim=0)


class PrimSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "primitive"

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(4*14*14, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.softmax(out, dim=0)


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "controller"

        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear((2*14*14)+1, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, T):
        out = self.conv(x)
        out = torch.flatten(out)
        out = torch.cat((out, torch.tensor([T])), 0)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return torch.sigmoid(out)
