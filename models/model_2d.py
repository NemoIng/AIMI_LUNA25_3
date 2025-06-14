import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34Base(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNet34Base, self).__init__()
        # Load pretrained ResNet34
        self.resnet34 = models.resnet34(weights=weights)
        
        # Replace the fully connected layer with a custom classification layer
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.resnet34(x)

class ResNet34(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1', dropout=0.3):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1', dropout=[0.3]):
        super(ResNet34, self).__init__()
        self.resnet34 = models.resnet34(weights=None)
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 1)
            nn.Identity(),               # fc.0 → dummy om de index te schuiven
            nn.Linear(num_features, 256),# fc.1
            nn.ReLU(),                   # fc.2
            nn.Dropout(p=dropout[0]),    # fc.3
            nn.Linear(256, 1)            # fc.4
        )



    def forward(self, x):
        return self.resnet34(x)


class ResNet34_exp(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1', dropout=0.3, batchnorm=True):
        super(ResNet34_exp, self).__init__()
        # Load pretrained ResNet34
        self.resnet34 = models.resnet34(weights=None)
        self.dropout = dropout
        self.batchnorm = batchnorm
        
        # Replace the fully connected layer with a custom classification layer
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256) if self.batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
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