#!/usr/bin/env python3
import argparse

from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import BVAE

def set_random_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', default=1, type=int, help="Random seed")
  parser.add_argument('--epochs', default=1e6, type=float, help="Number of epochs for training")
  parser.add_argument('--batch_size', default=64, type=int, help="Batch size")

  parser.add_argument('--beta', default=4, type=float, help="Beta parameter for KL-divergence")
  parser.add_argument('--gamma', default=1000, type=float, help="Gamma parameter for KL-divergence")
  parser.add_argument('--learning_rate', '-lr', dest='lr', default=1e-4, type=float, help="Learning rate")
  parser.add_argument('--beta_1', default=0.9, type=float, help="Adam optimiser beta 1")
  parser.add_argument('--beta_2', default=0.999, type=float, help="Adam optimiser beta 2")

  args = parser.parse_args()
  
  set_random_seed(args.seed)
  
  net = BVAE(in_channels=3, z=10)
  optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))

  transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()])
  
  dataset = ImageFolder(root='../data/celeba', transform=transform)
  data_loader = DataLoader(dataset,
                           batch_size=args.batch_size, 
                           shuffle=True, 
                           pin_memory=True,
                           drop_last=True)
  
  net.train()
  for epoch in range(int(args.epochs)):
    t = tqdm(enumerate(data_loader), total=int(len(data_loader.dataset) / data_loader.batch_size))
    t.set_description(f"Epoch {epoch}/{int(args.epochs)}")
    for i, (imgs, labels) in t:
      x = Variable(imgs)
      reconstruction, mu, log_var = net(x)
      
      # reconstruction loss using gaussian
      reconstruction = torch.sigmoid(reconstruction)
      MSE = F.mse_loss(reconstruction, x, reduction='sum').div(args.batch_size)

      # kl divergence
      batch_size, _ = mu.size()
      assert batch_size != 0
      
      if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
      if log_var.data.ndimension == 4:
        log_var = log_var.view(log_var.size(0), log_var.size(1))

      KLD = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
      TKLD = KLD.sum(1).mean(0, True) # Total KLD
      DWKLD = KLD.mean(0) # Dimension Wise KLD
      MKLD = KLD.mean(1).mean(0, True) # Mean KLD

      loss = MSE + args.beta * TKLD

      optim.zero_grad()
      loss.backward()
      optim.step()

      t.set_postfix({"Recon Loss": str(round(MSE.item(), 2)), "KLD": str(round(TKLD.item(), 4))})
       
