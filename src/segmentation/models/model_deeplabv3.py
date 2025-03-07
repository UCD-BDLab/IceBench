import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 , resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP


class DeepLabV3(nn.Module):
    def __init__(self, weights=None, frozen_start=False, in_channel = 24 , num_classes=21,  gamma=2.0, alpha=0.5):
        super(DeepLabV3, self).__init__()

        self.encoder = resnet18(weights=weights)
        self.encoder = IntermediateLayerGetter(self.encoder, return_layers={"layer2": "out"})

        if frozen_start:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.decoder = nn.Sequential(
            ASPP(in_channels=128, atrous_rates=[12, 24, 36], out_channels=128),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )

        avgpool_replacer = nn.AvgPool2d(8, 8)
        if isinstance(self.decoder[0].convs[-1][0], nn.AdaptiveAvgPool2d):
            self.decoder[0].convs[-1][0] = avgpool_replacer
        else:
            print('Check the model! Is there an AdaptiveAvgPool2d somewhere?')

        self.encoder.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)


    def forward(self, x):
        input_shape = x.shape[-2:]

        features = self.encoder(x)['out']

        logits = self.decoder(features)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)

        return logits





