import torch
import torch.nn as nn


class CrossNormalization(nn.Module):

    """
        A cross-normalization layer intended to replace the zero-convolution
        typically employed when fine-tuning models with additional inputs.
        The idea and implementation follows: https://arxiv.org/pdf/2408.06070
    """

    def __init__(self,
                 dims,
                 eps=1e-5,
                 ):
        super().__init__()
        self.dims = dims
        self.eps = eps

        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def forward(self, x, x_ref=None):

        if not self._enabled:
            return x
        
        x_ref = x if x_ref is None else x_ref
        mean_ref = torch.mean(x_ref, dim=self.dims, keepdim=True)
        std_ref = torch.std(x_ref, dim=self.dims, keepdim=True)

        mean_x = torch.mean(x, dim=self.dims, keepdim=True)
        std_x = torch.std(x, dim=self.dims, keepdim=True)

        x = (x - mean_x) * (std_ref / (std_x + 1e-5)) + mean_ref

        return x