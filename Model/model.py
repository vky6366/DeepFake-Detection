import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
    
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1184, 64)  # Adjust depending on MFCC shape
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)