import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, out_dim, hidden_sizes, activation=nn.Identity, gpu_id=0):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [out_dim], activation)

    def forward(self, obs, act):
        out = self.net(torch.cat([obs, act], dim=-1))
        return torch.squeeze(out, -1) # Critical to ensure q has right shape.

class CNNFunction(nn.Module):   
    def __init__(self, obs_dim, act_dim, out_dim, hidden_sizes, activation=nn.ReLU,
                 xavier=True, gpu_id=0):    
        super().__init__()
        self.act_dim = act_dim
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        self.base_out_channels = 16
        
        self.conv0 = nn.Conv3d(
            in_channels=4,
            out_channels=self.base_out_channels,
            kernel_size=(5, 5, 5),
            padding=1).to(self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(
            in_channels=self.base_out_channels,
            out_channels=self.base_out_channels,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=self.base_out_channels,
            out_channels=self.base_out_channels*2,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=self.base_out_channels*2,
            out_channels=self.base_out_channels*2,
            kernel_size=(3, 3, 3),
            padding=0).to(
            self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.Linear(in_features=self.base_out_channels*16, out_features=256).to(self.device)
        self.prelu4 = nn.PReLU().to(self.device)
        self.fc2 = nn.Linear(in_features=256+act_dim, out_features=128).to(self.device)
        self.prelu5 = nn.PReLU().to(self.device)
        self.fc3 = nn.Linear(in_features=128, out_features=out_dim).to(self.device)

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, obs, act):
        obs = obs.reshape(-1, 4, 45, 45, 45).to(self.device) / 255.0
        x = self.conv0(obs)
        x = self.prelu0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.view(-1, self.base_out_channels*16)
        x = self.fc1(x)
        x = self.prelu4(torch.cat([x, act.reshape(-1, self.act_dim)], axis=-1))
        x = self.fc2(x)
        x = self.prelu5(x)
        x = self.fc3(x)
        return x

class MLPSquashedGaussian(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std


    




