import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from Utils.face_regions import FacialRegionAnalyzer

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.facial_analyzer = FacialRegionAnalyzer()

        # Hook to capture gradients
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output  # Store the activation maps

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # Store the gradients

    def generate_heatmap(self, input_tensor, target_class=None):
        """Runs a forward + backward pass to compute Grad-CAM heatmap."""
        
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Compute gradients for the target class
        target_score = output[:, target_class]
        target_score.backward()

        # Compute Grad-CAM heatmap
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global Average Pooling over spatial dims
        activation = self.activations  # Get feature maps

        heatmap = (activation * gradients).sum(dim=1).squeeze()  # Weighted sum of feature maps
        heatmap = F.relu(heatmap)  # Remove negative values
        heatmap = heatmap.cpu().detach().numpy()

        # Normalize heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5)
        return heatmap

    def apply_heatmap(self, original_img, heatmap):
        """Overlays the heatmap onto the original image."""
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))  # Resize to match input
        heatmap = np.uint8(255 * heatmap)  # Convert to uint8 for colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap

        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)  # Blend original + heatmap
        return overlay

    def analyze_facial_regions(self, original_img, heatmap):
        """Analyzes which facial regions the model is focusing on."""
        # Get facial landmarks
        landmarks = self.facial_analyzer.get_landmarks(original_img)
        if landmarks is None:
            return None, None

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Get region scores
        region_scores = self.facial_analyzer.analyze_heatmap_regions(
            original_img, heatmap_resized, landmarks
        )
        
        # Get focused regions
        focused_regions = self.facial_analyzer.get_focused_regions(region_scores)
        
        return region_scores, focused_regions