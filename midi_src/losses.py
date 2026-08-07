import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        log_pt = self.ce(pred, target)
        pt = torch.exp(-log_pt)
        if isinstance(self.alpha, torch.Tensor):
            # clamp to avoid out-of-bounds on ignore_index positions
            alpha_t = self.alpha[target.clamp(min=0)]
        else:
            alpha_t = self.alpha
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            mask = log_pt != 0
            return focal_loss[mask].mean() if mask.any() else focal_loss.sum()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

