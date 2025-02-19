import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.base_model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()  # ✅ Ensure output is in range (0,1)
        )

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)  # ✅ Now outputs probability (0 to 1)
