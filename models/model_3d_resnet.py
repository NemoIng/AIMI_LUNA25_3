import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class ResNet3D(nn.Module):
    def __init__(self, num_classes, input_channels=3, pretrained=True, freeze_bn=False, dropout=[0.0, 0.0]):
        super(ResNet3D, self).__init__()
        self.model = r3d_18(pretrained=pretrained)

        # Modify the first conv layer to accept custom input channels
        if input_channels != 3:
            self.model.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                                           stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # Replace the final classification layer
        if dropout > 0:
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(256, 1)
            )
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)