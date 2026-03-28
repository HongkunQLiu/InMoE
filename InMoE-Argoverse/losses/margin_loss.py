import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxMarginLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, predictions, targets):
        # predictions: [B, H, K] - predicted probabilities
        # targets: [B, H] - ground truth mode indices (argmin over distance)

        assert predictions.dim() == 2 and targets.dim() == 1
        assert predictions.size(0) == targets.size(0)

        N, K = predictions.shape
        gt_prob = predictions.gather(1, targets.unsqueeze(1))  # [N, 1]
        margin_loss = self.margin + predictions - gt_prob  # [N, K]
        margin_loss.scatter_(1, targets.unsqueeze(1), 0.0)
        margin_loss = torch.clamp(margin_loss, min=0.0)

        return margin_loss.mean()
