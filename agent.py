import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    def __init__(self, start_pos):
        self.pos = start_pos


class Primitive(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        self.type = "primitive"
        modules = build_modules(in_dim, out_dim, hidden_dims)
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        out = self.network(x)
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


def build_modules(in_dim, out_dim, hidden_dims):
    modules = [nn.Linear(in_dim, hidden_dims[0]),
               nn.ReLU()]
    for i in range(1, len(hidden_dims)):
        modules.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        modules.append(nn.ReLU)
    modules.append(nn.Linear(hidden_dims[-1], out_dim))
    return modules