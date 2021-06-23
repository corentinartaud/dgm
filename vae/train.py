import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import argparse
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

import model

matplotlib.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epoch for VAE to train")
args = parser.parse_args()

epochs = args.epochs
batch_size = 64
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([transforms.ToTensor(),])

# train and validation data
train_data = datasets.MNIST(
    root='../data',
    train=True,
    download=True,
    transform=transform
)

val_data = datasets.MNIST(
    root='../data',
    train=False,
    download=True,
    transform=transform
)

# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

model = model.LVAE().to(device)
opt = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

def final_loss(bce_loss, mu, log_var):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def fit(model, data_loader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(data_loader)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        opt.zero_grad()
        reconstruction, mu, log_var = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, log_var)
        running_loss += loss.item()
        loss.backward()
        opt.step()

    train_loss = running_loss / len(data_loader.dataset)
    return train_loss

def validate(model, data_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=int(len(val_data) / data_loader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, log_var = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, log_var)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data) / data_loader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f"output/output{epoch}.png", nrow=num_rows)
        
        val_loss = running_loss / len(data_loader.dataset)
        return val_loss

train_loss = []
val_loss = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
