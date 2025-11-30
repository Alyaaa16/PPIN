# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

class KpLoss(nn.Module):
    """
    Keypoint heatmap regression loss.
    Expected inputs:
      - pred: (N, C, H, W)  predicted heatmaps (real values, typically sigmoid or raw)
      - target: (N, C, H, W) ground-truth heatmaps (0..1)
      - loss_mask: (N, C) or (N,) tensor of 0/1 indicating which keypoints exist per sample
    Behavior:
      - Computes elementwise MSE between pred and target
      - Weighs each channel by loss_mask (per-keypoint presence)
      - Normalizes by the number of valid keypoints * H * W
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, loss_mask=None):
        """
        pred: Tensor (N,C,H,W)
        target: Tensor (N,C,H,W)
        loss_mask: Tensor (N,C) or (N,) or None
        """
        if not (pred.shape == target.shape):
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

        N, C, H, W = pred.shape

        # element-wise squared error
        diff = (pred - target) ** 2  # (N,C,H,W)

        # If loss_mask provided, apply it per channel
        if loss_mask is None:
            # average over all elements
            loss = diff.sum()
            if self.reduction == 'mean':
                loss = loss / (N * C * H * W + EPS)
            return loss
        else:
            # Ensure loss_mask shape is (N, C)
            lm = loss_mask
            if lm.dim() == 1:
                # (N,) -> broadcast to (N,C)
                lm = lm.unsqueeze(1).expand(-1, C)
            elif lm.dim() == 2 and lm.shape[1] != C:
                # if lm shape (N, num_landmarks) but C differs, try to broadcast if possible
                # otherwise raise
                if lm.shape[1] == C:
                    pass
                else:
                    raise ValueError(f"loss_mask second dim {lm.shape[1]} != pred channels {C}")
            lm = lm.to(dtype=pred.dtype, device=pred.device)  # (N,C)

            # expand to (N,C,1,1)
            lm_exp = lm.unsqueeze(-1).unsqueeze(-1)

            weighted_diff = diff * lm_exp  # zero out missing keypoints
            total_weight = lm.sum()  # number of valid keypoints across batch

            # Sum over pixels and channels
            loss_sum = weighted_diff.sum()

            # Normalize by valid pixels. If no valid keypoint, avoid divide-by-zero by normalizing by N*C*H*W.
            if total_weight.item() > 0:
                normalizer = total_weight * (H * W)
            else:
                normalizer = N * C * H * W

            if self.reduction == 'mean':
                return loss_sum / (normalizer + EPS)
            else:
                return loss_sum


class CLALoss(nn.Module):
    """
    Classification loss for per-keypoint binary labels.
    Expected inputs:
      - pred_logits: (N, C) raw logits (no sigmoid)
      - labels: (N, C) binary labels (0/1)
      - loss_mask: (N, C) or (N,) mask to ignore missing keypoints
    Behavior:
      - Uses BCEWithLogitsLoss elementwise (reduction='none')
      - Applies loss_mask to ignore missing keypoints
      - Averages over valid entries
    """
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        # We'll use BCEWithLogitsLoss per-element with no reduction and handle masking
        self.pos_weight = pos_weight

    def forward(self, pred_logits, labels, loss_mask=None):
        """
        pred_logits: Tensor (N, C)
        labels: Tensor (N, C)
        loss_mask: Tensor (N, C) or (N,)
        """
        if not (pred_logits.shape == labels.shape):
            raise ValueError(f"pred and labels must have same shape, got {pred_logits.shape} vs {labels.shape}")

        device = pred_logits.device
        labels = labels.to(device=device, dtype=pred_logits.dtype)

        # elementwise BCE with logits
        if self.pos_weight is not None:
            # pos_weight expected to be 1D tensor of length C or scalar
            bce = F.binary_cross_entropy_with_logits(pred_logits, labels, pos_weight=self.pos_weight.to(device), reduction='none')
        else:
            bce = F.binary_cross_entropy_with_logits(pred_logits, labels, reduction='none')  # (N,C)

        if loss_mask is None:
            return bce.mean()

        lm = loss_mask
        if lm.dim() == 1:
            lm = lm.unsqueeze(1).expand_as(bce)  # (N,C)
        elif lm.dim() == 2 and lm.shape != bce.shape:
            # attempt to broadcast if possible
            if lm.shape[0] == bce.shape[0] and lm.shape[1] == bce.shape[1]:
                pass
            else:
                raise ValueError(f"loss_mask shape {lm.shape} incompatible with predictions {bce.shape}")
        lm = lm.to(device=device, dtype=bce.dtype)

        weighted_bce = bce * lm
        total_weight = lm.sum()

        if total_weight.item() > 0:
            return weighted_bce.sum() / (total_weight + EPS)
        else:
            # fallback to mean to avoid NaN
            return weighted_bce.mean()
