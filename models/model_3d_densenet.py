import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

class DenseNet3D(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, dropout=0.3):
        super(DenseNet3D, self).__init__()
        self.model = DenseNet121(
            spatial_dims=3,          # 3D volumes
            in_channels=input_channels,
            out_channels=num_classes,
            dropout_prob=dropout
        )

    def forward(self, x):
        return self.model(x)