#https://github.com/nansencenter/s1_icetype_cnn/blob/v1.1/cnn_for_sea_ice_types.ipynb


import torch.nn as nn
import torch
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, input_channels, n_outputs, pretrained_model=None):
        self.dropout_rate = 0.1
        super(CNN, self).__init__()
        if pretrained_model is None:
            self.pretrained = None
        else:
            self.pretrained = pretrained_model

        # Convolutions
        self.c1 = nn.Conv2d(input_channels,32,kernel_size=3,stride=1, padding=0)
        self.c2 = nn.Conv2d(32,32,kernel_size=3,stride=1, padding=0)
        self.c3 = nn.Conv2d(32,32,kernel_size=3,stride=1, padding=0)

        # Linear
        self.l_c1 = nn.Linear(32,n_outputs)

        # Batch Norm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)

        

    def forward(self, x):
        
        if self.pretrained is not None:
            x = self.pretrained(x)

        x = self.c1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.c3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.avg_pool2d(x, kernel_size=x.data.shape[2])
        
        x = x.view(x.size(0), x.size(1))
        x = self.l_c1(x)

        return x

