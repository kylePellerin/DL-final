import torch 
import torch.nn as nn 
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class BowdoinClassifier(nn.Module):
    def __init__(self, num_classes=4, num_countries=48, num_majors=42, 
                 dropout=0.4, freeze_backbone=True):
        super().__init__()

        # Backbone from EfficientNet B3
        weights  = EfficientNet_B3_Weights.IMAGENET1K_V1
        backbone = efficientnet_b3(weights=weights)

        self.backbone = backbone.features # (B, 1536, H', W')
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # (B, 1536, 1, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Fine-tune last two blocks
            for param in self.backbone[-2:].parameters():
                param.requires_grad = True

        # Shared Neck for Classifier 
        self.neck = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout), 
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

        # Task Heads 
        self.class_head = nn.Linear(256, num_classes) # 4 -> one-hot -> CrossEntropy
        self.country_head = nn.Linear(256, num_countries) # 48 -> multi-hot -> CrossEntropy
        self.major_head = nn.Linear(256, num_majors) # 42 -> multi-hot -> BCEWithLogits

    def forward(self, x):
        x = self.backbone(x) # (B, 1536, H', W')
        x = self.pool(x).squeeze(-1).squeeze(-1) # (B, 1536)
        features = self.neck(x) # (B, 256)

        class_logits = self.class_head(features) # (B, 4)
        country_logits = self.country_head(features) # (B, 48)
        major_logits = self.major_head(features) # (B, 42)

        return class_logits, country_logits, major_logits
                

