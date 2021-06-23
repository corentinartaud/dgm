import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_LAYERS = 512
FEATURES = 16

class LVAE(nn.Module):
  def __init__(self):
      super(LVAE, self).__init__()

      # encoder
      self.enc1 = nn.Linear(in_features=784, out_features=HIDDEN_LAYERS)
      self.enc2 = nn.Linear(in_features=HIDDEN_LAYERS, out_features=FEATURES * 2)

      # decoder
      self.dec1 = nn.Linear(in_features=FEATURES, out_features=HIDDEN_LAYERS)
      self.dec2 = nn.Linear(in_features=HIDDEN_LAYERS, out_features=784)

  def reparameterize(self, mu, log_var):
      """
      :param mu: from the encoder's latent space
      :param log_var: log variance from the encoder's latent space
      """
      std = torch.exp(0.5 * log_var) # standard deviation
      eps = torch.randn_like(std) # `randn_like` as we need the same size
      sample = mu + (eps * std) # sampling as if coming from the input space
      return sample

  def forward(self, x):
      # encoding
      x = F.relu(self.enc1(x))
      x = self.enc2(x).view(-1, 2, FEATURES)

      # get `mu` and `log_var`
      mu = x[:, 0, :] # the first features values as mean
      log_var = x[:, 1, :] # the other feature values

      z = self.reparameterize(mu, log_var)

      # decoding
      x = F.relu(self.dec1(z))
      reconstruction = torch.sigmoid(self.dec2(x))
      return reconstruction, mu, log_var
