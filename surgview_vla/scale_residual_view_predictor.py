"""Optional View Predictor variant with a frozen base and gated scale residual."""

from __future__ import annotations

import math

import torch
from torch import nn


class ScaleResidualViewPredictor(nn.Module):
    """Keep every base output except desired scale; predict scale as gated residual.

    The base model is permanently frozen.  Consequently current descriptors,
    desired center/depth, and their gradients cannot change in this variant.
    """

    def __init__(
        self,
        base: nn.Module,
        hidden_dim: int = 256,
        scale_residual_scale: float = 1.0,
        train_change_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if scale_residual_scale <= 0:
            raise ValueError("scale_residual_scale must be positive")
        if not 0 < train_change_rate < 1:
            raise ValueError("train_change_rate must be between zero and one")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        feature_dim = hidden_dim * 2
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.scale_residual = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.constant_(
            self.scale_gate[-1].bias,
            math.log(train_change_rate / (1.0 - train_change_rate)),
        )
        nn.init.zeros_(self.scale_residual[-1].weight)
        nn.init.zeros_(self.scale_residual[-1].bias)
        self.register_buffer(
            "scale_residual_scale", torch.tensor(float(scale_residual_scale), dtype=torch.float32)
        )

    def train(self, mode: bool = True) -> "ScaleResidualViewPredictor":
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        h_t: torch.Tensor,
        z_op: torch.Tensor,
        depth_t: torch.Tensor,
        gate_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.base(h_t, z_op, depth_t)
        feature = torch.cat((base["z_cur"].mean(1), base["z_des"].mean(1)), dim=-1)
        gate_logit = self.scale_gate(feature)[:, 0]
        gate_probability = gate_logit.sigmoid()
        residual_normalized = self.scale_residual(feature)[:, 0]
        residual = residual_normalized * self.scale_residual_scale
        gate = (
            gate_probability
            if gate_threshold is None
            else (gate_probability >= float(gate_threshold)).to(residual.dtype)
        )
        scale_delta = gate * residual
        g_cur = base["g_cur"].detach()
        g_des = base["g_des"].detach().clone()
        g_des[:, 2] = g_cur[:, 2] + scale_delta
        return {
            "z_cur": base["z_cur"].detach(),
            "z_des": base["z_des"].detach(),
            "g_cur": g_cur,
            "g_des": g_des,
            "e_view": g_des - g_cur,
            "scale_gate_logit": gate_logit,
            "scale_gate_probability": gate_probability,
            "scale_residual_normalized": residual_normalized,
            "scale_residual": residual,
            "scale_delta": scale_delta,
        }
