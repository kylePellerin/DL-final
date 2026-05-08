import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torch.load("./data/images")
data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

embedding_dim = 10 #what
num_class = 10 #what
latent_dim = 100 #what
img_shape = 600*800

class DCGAN_gen(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelembedding = nn.Embedding(num_class, embedding_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(110, 128, kernel_size=4, stride=1, padding=0),  # [batch, 128, 4, 4]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [batch, 64, 8, 8]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # [batch, 32, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=3),     # [batch, 1, 28, 28]
            nn.Tanh()
        )
    def forward(self, noise, labels):
        c = self.labelembedding(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(-1, -1, noise.size(2), noise.size(3))
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img

class DCGAN_disc(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelembedding = nn.Embedding(num_class, embedding_dim)

        self.model = nn.Sequential(
            nn.Conv2d(1 + embedding_dim, 64, kernel_size=4, stride=2, padding=1), # 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(), 
            nn.Linear(16*6*6, 1),            
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        c = self.labelembedding(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, c], 1)
        validity = self.model(x)
        return validity

generator_DCGAN = DCGAN_gen().to(device)
discriminator_DCGAN = DCGAN_disc().to(device)


import torch.optim as optim 

d_optimizer = optim.Adam(discriminator_DCGAN.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator_DCGAN.parameters(), lr=0.0002)
loss = nn.BCELoss()

def noise_2d(batch_size):
    n = torch.randn(batch_size, 100, 1, 1)
    return n.to(device)

def discriminator_train_step(real_data, real_labels, fake_data, fake_labels):
    vec_ones = torch.ones(len(real_data), 1).to(device)
    vec_zeros = torch.zeros(len(real_data), 1).to(device)

    discriminator_DCGAN.zero_grad()

    prediction_real = discriminator_DCGAN(real_data, real_labels)
    error_real = loss(prediction_real, vec_ones)
    error_real.backward()

    prediction_fake = discriminator_DCGAN(fake_data.detach(), fake_labels)
    error_fake = loss(prediction_fake, vec_zeros)
    error_fake.backward()

    d_optimizer.step()
    return error_real + error_fake


def generator_train_step(fake_data, fake_labels):
    vec_ones = torch.ones(len(fake_data), 1).to(device)

    g_optimizer.zero_grad()

    prediction = discriminator_DCGAN(fake_data, fake_labels)
    error = loss(prediction, vec_ones)
    error.backward()

    g_optimizer.step()
    return error


num_epochs = 40 
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    N = len(data_loader) 
    for _, (images, labels) in enumerate(data_loader):
        n_images = len(images)
        
        real_data = images.view(n_images, 1, 28, 28).to(device)
        real_labels = labels.to(device)
        z = noise_2d(n_images).to(device)
        fake_labels = torch.randint(0, num_class, (n_images,)).to(device)

        fake_data = generator_DCGAN(noise_2d(n_images), fake_labels).to(device)
        
        # fake_data = fake_data
        d_loss = discriminator_train_step(real_data, real_labels, fake_data, fake_labels)

        fake_labels = torch.randint(0, num_class, (n_images,)).to(device)
        fake_data = generator_DCGAN(noise_2d(n_images), fake_labels).to(device)
        g_loss = generator_train_step(fake_data, fake_labels)

    if (epoch+1) % 5 == 0:
        plot_samples_2d()
        print(f"Epoch: {epoch+1}")