import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability for the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class WeightedBCE(nn.Module):
    def  __init__(self, weight = [0.1,0.9]):
        super(WeightedBCE, self).__init__()  # Initialize the base class (nn.Module)
        self.weight = weight


    def forward(self,inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.weight[1], dtype=torch.float32))
        loss = bce_loss(inputs, targets)
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1.0  # To avoid division by zero
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if needed
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class JaccardLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(JaccardLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        # Apply sigmoid if the model output is logits
        if pred.max() > 1:  # This assumes that the model outputs logits (before sigmoid)
            pred = torch.sigmoid(pred)
        
        # Apply threshold to get binary predictions (0 or 1)
        pred = (pred > self.threshold).float()
        
        # Ensure target is also binary (0 or 1)
        target = target.float()  # Make sure target is float for computation
        
        # Compute intersection and union
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        
        # Compute Jaccard (IoU) loss (1 - Jaccard)
        jaccard_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
        return jaccard_loss
