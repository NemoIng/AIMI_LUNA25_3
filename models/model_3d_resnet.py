import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

class ResNet3D(nn.Module):
    def __init__(self, num_classes, input_channels=3, pretrained=True, freeze_bn=False, dropout=[0.0, 0.0]):
        super(ResNet3D, self).__init__()
        if pretrained:
            self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            self.model = r3d_18(weights=None)

        # Modify the first conv layer to accept custom input channels
        if input_channels != 3:
            self.model.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                                       stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # Replace the final classification layer
        if any(d > 0 for d in dropout):
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout[0] if len(dropout) > 0 else 0.0),
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
