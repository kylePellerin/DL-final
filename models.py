import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
condition_dim = 94 # 4 (Class) + 48 (Country) + 42 (Major)
channels = 3
img_size = 64 #resized down to make it easier but native is 600 X 800

# --- GENERATOR ---
class DCGAN_gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (batch, 194, 1, 1)
            nn.ConvTranspose2d(latent_dim + condition_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Size: (batch, 256, 4, 4)
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Size: (batch, 128, 8, 8)
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Size: (batch, 64, 16, 16)
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # Size: (batch, 32, 32, 32)
            
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Final Size: (batch, 3, 64, 64)
        )

    def forward(self, noise, combined_labels):
        c = combined_labels.view(combined_labels.size(0), combined_labels.size(1), 1, 1)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)


# --- DISCRIMINATOR ---
class DCGAN_disc(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels: 3 (Image) + 94 (Condition) = 97
        self.model = nn.Sequential(
            # Input: (batch, 97, 64, 64)
            nn.Conv2d(channels + condition_dim, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 64, 32, 32)
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 128, 16, 16)
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 256, 8, 8)
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 512, 4, 4)
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Final Size: (batch, 1, 1, 1)
        )

    def forward(self, img, combined_labels):
        c = combined_labels.view(combined_labels.size(0), combined_labels.size(1), 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, c], dim=1)
        validity = self.model(x)
        return validity.view(-1, 1) # flatten to (batch, 1)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid



