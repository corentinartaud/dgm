#!/usr/bin/env python3
from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.autograd import Variable

class BVAE(nn.Module):
  def __init__(self, in_channels=3, z=10):
    super(BVAE, self).__init__()
    self.in_channels = in_channels 
    self.z = z
    self.encoder = nn.Sequential(
      nn.Conv2d(self.in_channels, 32, 4, 2, 1),
      nn.ReLU(True),
      nn.Conv2d(32, 32, 4, 2, 1),
      nn.ReLU(True),
      nn.Conv2d(32, 64, 4, 2, 1),
      nn.ReLU(True),
      nn.Conv2d(64, 64, 4, 2, 1),
      nn.ReLU(True),
      nn.Conv2d(64, 256, 4, 1),
      nn.ReLU(True),
    ) 
    self.fc1 = nn.Linear(256, self.z * 2)
    self.fc2 = nn.Linear(self.z, 256)
    self.decoder = nn.Sequential(
      nn.ReLU(True),
      nn.ConvTranspose2d(256, 64, 4),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 64, 4, 2, 1),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 32, 4, 2, 1),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 32, 4, 2, 1),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, self.in_channels, 4, 2, 1),
    )
    
    # TODO: move this in its own function
    # weight init
    for m in self.modules():
      if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: m.bias.data.fill_(0)

  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = Variable(torch.randn_like(std)) 
    return mu + (std * eps)
 
  def forward(self, x):
    # encoder
    x = self.encoder(x)
    x = x.view((-1, 256 * 1 * 1))
    x = self.fc1(x)

    # reparameterize
    mu = x[:, :self.z]
    log_var = x[:, self.z:]
    z = self.reparameterize(mu, log_var)

    # decoder
    z = self.fc2(z)
    z = z.view((-1, 256, 1, 1))
    x_hat = self.decoder(z)
    return x_hat, mu, log_var

