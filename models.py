import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.utils.spectral_norm as spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
condition_dim = 94 # 4 (Class) + 48 (Country) + 42 (Major)
channels = 3
img_size = 128 #resized down to make it easier but native is 600 X 800

# --- GENERATOR ---
# class DCGAN_gen(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             # Input: (batch, 194, 1, 1)
#             nn.ConvTranspose2d(latent_dim + condition_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # Size: (batch, 256, 4, 4)
            
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # Size: (batch, 128, 8, 8)
            
#             nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # Size: (batch, 256, 16, 16)
            
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # Size: (batch, 128, 32, 32)

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # Size: (batch, 64, 64, 64)
            
#             nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.Tanh()
#             # Final Size: (batch, 3, 128, 128)
#         )

#     def forward(self, noise, combined_labels):
#         c = combined_labels.view(combined_labels.size(0), combined_labels.size(1), 1, 1)
#         x = torch.cat([noise, c], dim=1)
#         return self.model(x)

# class DCGAN_gen(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.model = nn.Sequential(
#         #     # Input: (batch, 194, 1, 1)
#         #     nn.ConvTranspose2d(latent_dim + condition_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(True),
#         #     # Size: (batch, 256, 4, 4)
            
#         #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(True),
#         #     # Size: (batch, 128, 8, 8)
            
#         #     nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(True),
#         #     # Size: (batch, 256, 16, 16)
            
#         #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(True),
#         #     # Size: (batch, 128, 32, 32)

#         #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(True),
#         #     # Size: (batch, 64, 64, 64)
            
#         #     nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
#         #     nn.Tanh()
#         #     # Final Size: (batch, 3, 128, 128)
#         # )
#         self.model = nn.Sequential(
#             # Input: (batch, 194, 1, 1)


#             nn.ConvTranspose2d(latent_dim + condition_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # Size: (batch, 256, 4, 4)

#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(128, affine=True),  # FIX 1: was BatchNorm2d
#             nn.LeakyReLU(0.2, inplace=True),
#             # Size: (batch, 128, 8, 8)

#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(64, affine=True),  # FIX 1: was BatchNorm2d
#             nn.LeakyReLU(0.2, inplace=True),
#             # Size: (batch, 64, 16, 16)


#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(32, affine=True),  # FIX 1: was BatchNorm2d
#             nn.LeakyReLU(0.2, inplace=True),
#             # Size: (batch, 32, 32, 32)

#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(16),  # FIX 1: was BatchNorm2d
#             nn.LeakyReLU(0.2, inplace=True),
#             # Size: (batch, 16, 64, 64)

#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
#             # Size: (batch, 3, 128, 128)
        
#             nn.Tanh()
#             # Final Size: (batch, 3, 128, 128)
#         )

#     def forward(self, noise, combined_labels):
#         c = combined_labels.view(combined_labels.size(0), combined_labels.size(1), 1, 1)
#         x = torch.cat([noise, c], dim=1)
#         return self.model(x)


# # --- DISCRIMINATOR ---
# class DCGAN_disc(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Input channels: 3 (Image) + 94 (Condition) = 97
#         self.model = nn.Sequential(
#             # Input: (batch, 97, 128, 128)
#             nn.Conv2d(channels + condition_dim, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             # Size: (batch, 64, 64, 64)
            
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             # Size: (batch, 128, 32, 32)
            
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             # Size: (batch, 256, 16, 16)
            
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             # Size: (batch, 512, 8, 8)

#             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             # Size: (batch, 512, 4, 4)
            
#             nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#             # Final Size: (batch, 1, 1, 1)
#         )

#     def forward(self, img, combined_labels):
#         c = combined_labels.view(combined_labels.size(0), combined_labels.size(1), 1, 1)
#         c = c.expand(-1, -1, img.size(2), img.size(3))
#         x = torch.cat([img, c], dim=1)
#         validity = self.model(x)
#         return validity.view(-1, 1) # flatten to (batch, 1)


class DCGAN_gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + condition_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),  
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),  
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),  
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),  
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, combined_labels):
        c = combined_labels.view(combined_labels.size(0), combined_labels.size(1), 1, 1)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)


class DCGAN_disc(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.condition_project = nn.Sequential(
            spectral_norm(nn.Linear(condition_dim, 16)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64 + 16, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False))
        )

    def forward(self, img, combined_labels):
        x = self.conv1(img)
        c_proj = self.condition_project(combined_labels) 
        c = c_proj.view(c_proj.size(0), c_proj.size(1), 1, 1)
        c = c.expand(-1, -1, x.size(2), x.size(3))
        
        x = torch.cat([x, c], dim=1)
        validity = self.conv2(x)
        return validity.view(-1, 1)
