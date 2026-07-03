import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Linear):
    def forward(self, x):
        # Optional: layer norm or absmax scaling on input
        x_norm = ...  # often with SubLN or similar in BitNet
        
        # Quantize weights to {-1, 0, +1} with STE
        w = self.weight
        w_scale = w.abs().mean()  # or median, etc.
        w_q = (w / w_scale).round().clamp_(-1, 1)   # ternary
        w_q = w_q + (w - w_q).detach()  # STE trick
        
        # Similar for activations if using low-bit acts
        return F.linear(x_norm, w_q * w_scale)