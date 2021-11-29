import torch
import torch.nn as nn
import torch.nn.functional as F


class Primitive(nn.Module):
    def __init__(self, out_dim, D):
        super().__init__()
        self.type = "primitive"
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.full = nn.Linear(4*(D-2)*(D-2), out_dim)

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out)
        out = self.full(out)
        return F.softmax(out, dim=0)


class Controller(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        self.type = "controller"
        modules = build_modules(in_dim, out_dim, hidden_dims)
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        out = self.network(x)
        return F.sigmoid(out)


class Reward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        self.type = "reward_fn"
        modules = build_modules(in_dim, out_dim, hidden_dims)
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


def build_modules(in_dim, out_dim, hidden_dims):
    modules = [nn.Linear(in_dim, hidden_dims[0]),
               nn.ReLU()]
    for i in range(1, len(hidden_dims)):
        modules.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_dims[-1], out_dim))
    return modules