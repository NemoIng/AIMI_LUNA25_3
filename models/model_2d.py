import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNet34, self).__init__()
        # Load pretrained ResNet34
        self.resnet34 = models.resnet34(weights=weights)
        
        # Replace the fully connected layer with a custom classification layer
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet34(x)

# To test the model definition:
if __name__ == "__main__":
    image = torch.randn(4, 3, 64, 64)

    model = ResNet34()

    # input image to model
    output = model(image)