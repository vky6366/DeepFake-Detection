import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetectorb5(nn.Module):
    def __init__(self):
        super(DeepfakeDetectorb5, self).__init__()
        self.base_model = models.efficientnet_b5(weights='IMAGENET1K_V1')
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)
    
    
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.base_model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

class AudioCNN(torch.nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 32 * 4, 2)  # Example size; change based on model

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)