import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
import torch

class CustomVGG16(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=None):
        super(CustomVGG16, self).__init__()
        # Load the pre-trained VGG16 model
        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify the first convolutional layer to have the correct number of input channels
        original_conv1 = self.vgg16.features[0]
        new_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, 
                              stride=original_conv1.stride, padding=original_conv1.padding)
        self.vgg16.features[0] = new_conv1
        
        # Update the classifier to have the correct number of output classes
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vgg16(x)


