from FocalLoss import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        # Sigmoid to get probabilities
        probs = torch.sigmoid(preds)
        targets = targets.float()

        # Ensure batch is first, then flatten everything else
        dims = tuple(range(1, preds.ndim))

        tp = torch.sum(probs * targets, dims)
        fp = torch.sum(probs * (1 - targets), dims)
        fn = torch.sum((1 - probs) * targets, dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = torch.pow((1 - tversky), self.gamma)
        return loss.mean()
