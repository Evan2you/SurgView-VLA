"""View Predictor: current and desired four-dimensional view regression.

The model consumes frozen semantic, operative and monocular-depth features.  It
predicts the current and desired view descriptors independently; ``E_view`` is
always their exact difference and is never represented by a separate head.
"""

from __future__ import annotations

import torch
from torch import nn


class _ViewQueryCore(nn.Module):
    """Small query decoder kept local so Stage II-B has no StarVLA dependency."""

    def __init__(self, hidden_dim: int, num_view_tokens: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.current_queries = nn.Parameter(torch.empty(1, num_view_tokens, hidden_dim))
        self.desired_queries = nn.Parameter(torch.empty(1, num_view_tokens, hidden_dim))
        nn.init.normal_(self.current_queries, std=0.02)
        nn.init.normal_(self.desired_queries, std=0.02)
        self.current_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.desired_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.current_norm = nn.LayerNorm(hidden_dim)
        self.desired_norm = nn.LayerNorm(hidden_dim)
        self.current_descriptor = nn.Linear(hidden_dim, 4)
        self.desired_descriptor = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        operative_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = torch.cat((semantic_tokens, operative_tokens, depth_tokens), dim=1)
        batch = context.shape[0]
        q_cur = self.current_queries.expand(batch, -1, -1)
        q_des = self.desired_queries.expand(batch, -1, -1)
        z_cur = self.current_norm(self.current_attn(q_cur, context, context)[0] + q_cur)
        z_des = self.desired_norm(self.desired_attn(q_des, context, context)[0] + q_des)
        return {
            "z_cur": z_cur,
            "z_des": z_des,
            "g_cur": self.current_descriptor(z_cur.mean(dim=1)),
            "g_des": self.desired_descriptor(z_des.mean(dim=1)),
        }


class ViewPredictor(nn.Module):
    """Map ``H_t + Z_op + depth_t`` to 4-D current/desired view states."""

    def __init__(
        self,
        h_dim: int = 4096,
        operative_dim: int = 1024,
        hidden_dim: int = 256,
        num_view_tokens: int = 4,
        num_heads: int = 8,
        view_mean: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        view_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.semantic_projector = nn.Sequential(
            nn.LayerNorm(h_dim), nn.Linear(h_dim, hidden_dim), nn.GELU()
        )
        self.operative_projector = nn.Sequential(
            nn.LayerNorm(operative_dim), nn.Linear(operative_dim, hidden_dim), nn.GELU()
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.core = _ViewQueryCore(hidden_dim, num_view_tokens, num_heads)
        mean = torch.as_tensor(view_mean, dtype=torch.float32)
        scale = torch.as_tensor(view_scale, dtype=torch.float32)
        if mean.shape != (4,) or scale.shape != (4,) or torch.any(scale <= 0):
            raise ValueError("view_mean/view_scale must be four values and scale must be positive")
        self.register_buffer("view_mean", mean)
        self.register_buffer("view_scale", scale)

    def forward(
        self,
        h_t: torch.Tensor,
        z_op: torch.Tensor,
        depth_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if h_t.ndim != 2 or h_t.shape[-1] != 4096:
            raise ValueError(f"H_t must have shape [B,4096], got {tuple(h_t.shape)}")
        if z_op.ndim != 2 or z_op.shape[-1] != 1024:
            raise ValueError(f"Z_op must have shape [B,1024], got {tuple(z_op.shape)}")
        if depth_t.ndim != 3 or tuple(depth_t.shape[-2:]) != (45, 80):
            raise ValueError(f"depth_t must have shape [B,45,80], got {tuple(depth_t.shape)}")
        semantic_tokens = self.semantic_projector(h_t.float()).unsqueeze(1)
        operative_tokens = self.operative_projector(z_op.float()).unsqueeze(1)
        depth_tokens = self.depth_encoder(depth_t.float().unsqueeze(1)).flatten(2).transpose(1, 2)
        normalized = self.core(semantic_tokens, operative_tokens, depth_tokens)
        g_cur = normalized["g_cur"] * self.view_scale + self.view_mean
        g_des = normalized["g_des"] * self.view_scale + self.view_mean
        return {
            "z_cur": normalized["z_cur"],
            "z_des": normalized["z_des"],
            "g_cur_normalized": normalized["g_cur"],
            "g_des_normalized": normalized["g_des"],
            "g_cur": g_cur,
            "g_des": g_des,
            "e_view": g_des - g_cur,
        }
