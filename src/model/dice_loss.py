import torch

from torchmetrics.segmentation import DiceScore

class DiceLoss(torch.nn.Module):
    
    def __init__(self, n_classes, mode='micro', ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.mode = mode
        
    def forward(self, logits, targets, smooth=1e-6):
        
        C = logits.shape[1]


        class_indices = torch.arange(C, device=logits.device).view(1, C, 1, 1)  # [1, C, 1, 1]
        targets_expanded = targets.unsqueeze(1)  # [B, 1, H, W]
        target_one_hot = (targets_expanded == class_indices).float() 
        preds = torch.softmax(logits, dim=1)

        # maskout ignore index
        valid = targets != self.ignore_index
        valid_expanded = valid.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, H, W]
        preds_masked = preds * valid_expanded
        target_one_hot_masked = target_one_hot * valid_expanded

        intersection = (preds_masked * target_one_hot_masked).sum(dim=(0, 2, 3))
        union = (preds_masked + target_one_hot_masked).sum(dim=(0, 2, 3))

        if self.mode == 'micro':
            intersection = intersection.sum()
            union = union.sum()

        dice_score = (2.0 * intersection + smooth) / (union + smooth)

        if self.mode == 'macro':
            dice_score = dice_score.mean()
        
        assert torch.isnan(dice_score) == False, "Dice score is NaN" 
        assert torch.isinf(dice_score) == False, "Dice score is Inf"

        return 1 - dice_score
