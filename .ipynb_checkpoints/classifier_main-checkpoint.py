import torch
import torch.nn as nn
import torch.optim as optim
import os

import torchvision.transforms as T
from torch.utils.data import DataLoader

from data import load_data, process_data, BowdoinData
from classifier_models import BowdoinClassifier

"""Global Variables"""
# Data Paths
image_dir = "./data/images/*"
csv_path = "./data/data_info.csv"

num_epochs = 200
batch_size = 5

save_dir = "./Output_Classifier/"
os.makedirs(save_dir, exist_ok=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Dataset 
data_dict = load_data(image_dir, csv_path)
transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5), #data augmentation
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), #data augmentation
        T.RandomRotation(degrees=7), #data augmentation
        T.RandomPerspective(distortion_scale=0.05, p=0.5), #data augmentation
        T.ToTensor(),
        T.Normalize((0.5173, 0.4501, 0.4103), (0.2840, 0.2643, 0.2671))
])

processed_data_dict = process_data(data_dict, device, transforms)
dataset = BowdoinData(processed_data_dict)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss Functions
ce_loss  = nn.CrossEntropyLoss()       # for class and country
bce_loss = nn.BCEWithLogitsLoss()      # for major (multi-hot)

def compute_loss(pred_class, pred_country, pred_major,
                 true_class, true_country, true_major):

    # CrossEntropy expects class indices, not one-hot
    # Convert one-hot → index with argmax
    loss_class   = ce_loss(pred_class,   true_class.argmax(dim=1))
    loss_country = ce_loss(pred_country, true_country.argmax(dim=1))
    loss_major   = bce_loss(pred_major,  true_major)   # stays multi-hot

    return loss_class + loss_country + loss_major

# Model, Optimizer, Scheduler
model = BowdoinClassifier().to(device)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for images, class_lbl, country_lbl, major_lbl in dataloader:
        images      = images.to(device)
        class_lbl   = class_lbl.to(device)
        country_lbl = country_lbl.to(device)
        major_lbl   = major_lbl.to(device)

        optimizer.zero_grad()

        pred_c, pred_co, pred_m = model(images)
        loss = compute_loss(pred_c, pred_co, pred_m,
                            class_lbl, country_lbl, major_lbl)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"bowdoin_classifier_epoch{epoch+1}.pth"))


# Final Model Save
torch.save(model.state_dict(), os.path.join(save_dir, "bowdoin_classifier.pth"))
print("✓ Final model saved.")