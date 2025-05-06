from FocalLoss import FocalLoss
import torch
import torch.nn as nn

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        smooth = 1.0
        dice = 1 - (2 * (probs * targets).sum() + smooth) / ((probs + targets).sum() + smooth)
        return (1 - self.dice_weight) * self.focal(inputs, targets) + self.dice_weight * dice
