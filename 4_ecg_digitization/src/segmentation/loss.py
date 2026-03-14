import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    Funkcja straty kombinująca stabilność Binary Cross-Entropy (BCE) 
    z precyzją przestrzenną Dice Loss, opisana w strategii Karbasiego:
    L = L_Dice + 0.5 * L_BCE
    Wymusza na sieci ignorowanie pogrubień atramentu i precyzyjne śledzenie cienkich linii sygnału.
    """
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: surowe logity (przed Sigmoid) z modelu
        # targets: maski binarne Ground Truth (0 lub 1)
        
        inputs = torch.sigmoid(inputs)
        
        # Wypłaszczenie tensorów
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 1. Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 2. Binary Cross Entropy Loss
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Całkowity koszt
        loss = dice_loss + self.bce_weight * bce
        
        return loss
