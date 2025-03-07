import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights



class ModifiedDenseNet121(nn.Module):
    def __init__(self, pretrained=True, input_channels=3, num_classes=1000):
        super(ModifiedDenseNet121, self).__init__()
        self.densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        
        if input_channels != 3:
            # Adjust the first layer to match input_channels
            self.densenet.features[0] = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the classifier to match num_classes
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
            
    def forward(self, x):
        return self.densenet(x)





