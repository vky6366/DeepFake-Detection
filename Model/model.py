import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.base_model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()  # Remove original classifier
        
        # ✅ Add a custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization for stable training
            nn.LeakyReLU(0.1),  # Prevent dead neurons
            nn.Dropout(p=0.6),  # Dropout to reduce overfitting
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            
            nn.Linear(256, 1)  # Binary classification output
        )

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)  # ✅ Now outputs probability (0 to 1)
