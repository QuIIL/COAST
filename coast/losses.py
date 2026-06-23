import torch
import torch.nn as nn


class SpotMSEPlusDiffMSELoss(nn.Module):
    def __init__(
        self,
        w_spot: float = 1.0,
        w_diff: float = 0.1,
        gene_mean=None,
        gene_var=None,
        gene_std=None,
        eps: float = 1e-8,
        norm_reg: bool = False,
        norm_diff: bool = False,
    ):
        super().__init__()
        self.w_spot = float(w_spot)
        self.w_diff = float(w_diff)
        self.gene_var = gene_var
        self.norm_reg = norm_reg
        self.norm_diff = norm_diff
        self.eps = eps

    def forward(
        self,
        y_hat_spot: torch.Tensor,
        y_true_spot: torch.Tensor,
        y_hat_diff: torch.Tensor,
        y_true_ctx: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spot_error = y_hat_spot - y_true_spot
        spot_loss = spot_error.square()
        if self.norm_reg:
            spot_loss = spot_loss / (self.gene_var + self.eps)
        spot_loss = spot_loss.mean()

        true_difference = y_true_spot[:, None, :] - y_true_ctx
        difference_error = y_hat_diff - true_difference
        difference_loss = difference_error.square()
        if self.norm_diff:
            difference_loss = difference_loss / (self.gene_var + self.eps)

        if pair_mask is not None:
            mask = pair_mask[:, :, None].to(difference_loss.dtype)
            denominator = mask.sum() * difference_loss.shape[-1] + self.eps
            difference_loss = (difference_loss * mask).sum() / denominator
        else:
            difference_loss = difference_loss.mean()

        total_loss = self.w_spot * spot_loss + self.w_diff * difference_loss
        return total_loss, spot_loss.detach(), difference_loss.detach()
