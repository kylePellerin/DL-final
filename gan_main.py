"""Main training loop for DCGAN on Bowdoin dataset"""
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T

from data import load_data, process_data, BowdoinData
from gan_models import DCGAN_gen, DCGAN_disc

"""Global Variables"""
# Data Paths
image_dir = "./data/images/*"
csv_path = "./data/data_info.csv"

latent_dim = 256
condition_dim = 94 # 4 (Class) + 48 (Country) + 42 (Major)
channels = 3
img_size = (800, 600)
num_epochs = 500
batch_size = 5

save_dir = "./Output_GAN/"
os.makedirs(save_dir, exist_ok=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Dataset 
data_dict = load_data(image_dir, csv_path)
transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5173, 0.4501, 0.4103), (0.2840, 0.2643, 0.2671))
])
processed_data_dict = process_data(data_dict, device, transforms)
dataset = BowdoinData(processed_data_dict)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Models
generator_DCGAN = DCGAN_gen().to(device)
discriminator_DCGAN = DCGAN_disc().to(device)

# Optimizers and Loss Function
d_optimizer = optim.Adam(discriminator_DCGAN.parameters(), lr=0.0001, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator_DCGAN.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss_fn = nn.BCEWithLogitsLoss()

# Noise Function
def noise_2d(batch_size):
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)

# Visual Function
def show_generated_images(save_path, generator, num_images=16):
    generator.eval() 
    
    with torch.no_grad():
        z = noise_2d(num_images)
        zeros_class = torch.zeros(num_images, 4, device=device)
        zeros_country = torch.zeros(num_images, 48, device=device)
        zeros_major = torch.zeros(num_images, 42, device=device)

        for i in range(num_images):
            zeros_class[i, torch.randint(0,  4, (1,)).item()] = 1.0
            zeros_country[i, torch.randint(0, 48, (1,)).item()] = 1.0
            zeros_major[i, torch.randint(0, 42, (1,)).item()] = 1.0

        fake_conditions = torch.cat([zeros_class, zeros_country, zeros_major], dim=1)
        fake_images = generator(z, fake_conditions)
        fake_images = (fake_images + 1) / 2.0
        
        grid = make_grid(fake_images.cpu(), nrow=4, padding=2, normalize=False)
        
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Faces")
        plt.imshow(grid.permute(1, 2, 0)) 

        # Save or Show
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:    
           plt.show() 
        
    generator.train() 

"""Begin Training Loop"""
for epoch in range(num_epochs):
    for i, (images, class_labels, country_labels, major_labels) in enumerate(dataloader):
        n_images = images.size(0)
        
        real_data = images.to(device)
        real_conditions = torch.cat([class_labels, country_labels, major_labels], dim=1).to(device)

        # Discriminator Training
        discriminator_DCGAN.zero_grad()
        
        pred_real = discriminator_DCGAN(real_data, real_conditions)
        loss_real = loss_fn(pred_real, torch.full((n_images, 1), 0.9, device=device)) 
        
        # Fake loss
        z = noise_2d(n_images)
        fake_data = generator_DCGAN(z, real_conditions)
        
        pred_fake = discriminator_DCGAN(fake_data.detach(), real_conditions)
        loss_fake = loss_fn(pred_fake, torch.full((n_images, 1), 0.1, device=device))
        
        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Generator Training
        generator_DCGAN.zero_grad()
        pred_fake_gen = discriminator_DCGAN(fake_data, real_conditions)
        g_loss = loss_fn(pred_fake_gen, torch.full((n_images, 1), 0.9, device=device))
        
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    if (epoch + 1) % 10 == 0:
        show_generated_images(f"{save_dir}epoch_{epoch+1}.png", generator_DCGAN, num_images=16)

# Save Model 
torch.save(generator_DCGAN.state_dict(), f"{save_dir}generator.pth")
torch.save(discriminator_DCGAN.state_dict(), f"{save_dir}discriminator.pth")



