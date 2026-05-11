import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 256
condition_dim = 94 # 4 (Class) + 48 (Country) + 42 (Major)
channels = 3
img_size = (800, 600)

# --- GENERATOR ---
class DCGAN_gen(nn.Module):
    def __init__(self, latent_dim=latent_dim, condition_dim=condition_dim, channels=channels):
        super().__init__()

        in_ch = latent_dim + condition_dim  # 350

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(out_c, affine=True),  # FIX 1: was BatchNorm2d
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            # (B, 350, 1, 1) → (B, 2048, 6, 4)
            nn.ConvTranspose2d(in_ch, 2048, kernel_size=(6, 4), stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(2048, affine=True),        # FIX 1
            nn.LeakyReLU(0.2, inplace=True),

            block(2048, 1024),  # → (B, 1024, 12, 8)
            block(1024,  512),  # → (B,  512, 24, 16)
            block( 512,  256),  # → (B,  256, 48, 32)
            block( 256,  128),  # → (B,  128, 96, 64)
            block( 128,   64),  # → (B,   64, 192, 128)
            block(  64,   32),  # → (B,   32, 384, 256)

            # Final → (B, 3, 768, 512) then interpolate to (800, 600)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, combined_labels):
        c = combined_labels.view(combined_labels.size(0), -1, 1, 1)
        x = torch.cat([noise, c], dim=1)
        x = self.model(x)                                          # (B, 3, 768, 512)
        x = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)  # (B, 3, 800, 600)
        return x


# --- DISCRIMINATOR ---
class DCGAN_disc(nn.Module):
    def __init__(self, channels=channels, condition_dim=condition_dim):
        super().__init__()

        in_ch = channels + condition_dim # 3 (image) + 94 (condition) = 97

        def block(in_c, out_c, bn=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False))]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.4))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(in_ch, 64, bn=False),   # (batch, 64, 400, 300)
            block(64, 128),               # (batch, 128, 200, 150)
            block(128, 256),              # (batch, 256, 100, 75)
            block(256, 512),              # (batch, 512, 50, 37)
            block(512, 512),              # (batch, 512, 25, 18)
            block(512, 512),              # (batch, 512, 12, 9)
            block(512, 512),              # (batch, 512, 6, 4)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.classifier = nn.Linear(512, 1)

    def forward(self, img, combined_labels):
        c = combined_labels.view(combined_labels.size(0), -1, 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3)) # (B, 94, 800, 600)
        x = torch.cat([img, c], dim=1) # (B, 97, 800, 600)

        x = self.features(x) # (B, 512, 6, 4)
        x = self.pool(x).view(x.size(0), -1) # (B, 512)
        validity = self.classifier(x) # (B, 1)
        return validity


# ── SANITY CHECK ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 2
    noise  = torch.randn(B, latent_dim, 1, 1).to(device)
    labels = torch.randn(B, condition_dim).to(device)

    G = DCGAN_gen().to(device)
    D = DCGAN_disc().to(device)

    fake  = G(noise, labels)
    score = D(fake.detach(), labels)

    assert fake.shape  == (B, 3, 800, 600), f"G output wrong: {fake.shape}"
    assert score.shape == (B, 1),           f"D output wrong: {score.shape}"
    print(f"✓ G output : {fake.shape}")
    print(f"✓ D output : {score.shape}")

    # And in your training loop, use this instead of BCELoss:
    # criterion = nn.BCEWithLogitsLoss()