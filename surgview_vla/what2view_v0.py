"""Lightweight FiLM-based What2View v0.

This module intentionally does not use cross-attention.  It consumes one pooled
language-conditioned semantic vector and one grid of spatial visual features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class What2ViewV0(nn.Module):
    """Predict an operative heatmap, operative visual token and visibility.

    ``mode`` is also used for the controlled ablations:

    * ``h_only``: semantic conditioning over a learned spatial canvas;
    * ``v_only``: spatial DINO features without semantic conditioning;
    * ``h_v``: the proposed semantic FiLM conditioning of spatial features.
    """

    MODES = ("h_only", "v_only", "h_v")

    def __init__(
        self,
        h_dim: int = 4096,
        v_dim: int = 1024,
        hidden_dim: int = 256,
        visual_hw: tuple[int, int] = (14, 24),
        heatmap_hw: tuple[int, int] = (45, 80),
        mode: str = "h_v",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.v_dim = v_dim
        self.visual_hw = visual_hw
        self.heatmap_hw = heatmap_hw

        self.semantic_projector = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(v_dim),
            nn.Linear(v_dim, hidden_dim),
        )
        self.position_grid = nn.Parameter(
            torch.empty(1, hidden_dim, visual_hw[0], visual_hw[1])
        )
        nn.init.normal_(self.position_grid, mean=0.0, std=0.02)
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_t: torch.Tensor | None,
        v_t: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.mode in ("h_only", "h_v") and h_t is None:
            raise ValueError(f"{self.mode} requires H_t")
        if self.mode in ("v_only", "h_v") and v_t is None:
            raise ValueError(f"{self.mode} requires V_t")

        reference = h_t if h_t is not None else v_t
        assert reference is not None
        batch = reference.shape[0]
        semantic = (
            self.semantic_projector(h_t.float())
            if h_t is not None and self.mode != "v_only"
            else self.position_grid.new_zeros(batch, self.position_grid.shape[1])
        )

        if v_t is not None and self.mode != "h_only":
            if tuple(v_t.shape[1:3]) != self.visual_hw:
                raise ValueError(
                    f"V_t spatial shape must be {self.visual_hw}, got {tuple(v_t.shape[1:3])}"
                )
            spatial = self.visual_projector(v_t.float()).permute(0, 3, 1, 2)
            spatial = spatial + self.position_grid
        else:
            spatial = self.position_grid.expand(batch, -1, -1, -1)

        if self.mode in ("h_only", "h_v"):
            gamma, beta = self.film(semantic).chunk(2, dim=-1)
            spatial = spatial * (1.0 + torch.tanh(gamma)[:, :, None, None])
            spatial = spatial + beta[:, :, None, None]

        low_resolution_logits = self.decoder(spatial)
        heatmap_logits = F.interpolate(
            low_resolution_logits,
            size=self.heatmap_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        pooled_spatial = spatial.mean(dim=(-2, -1))
        visibility_logit = self.visibility_head(
            torch.cat((semantic, pooled_spatial), dim=-1)
        )[:, 0]

        result = {
            "attention_logits": heatmap_logits,
            "attention": heatmap_logits.sigmoid(),
            "visibility_logit": visibility_logit,
            "visibility": visibility_logit.sigmoid(),
        }
        if v_t is not None and self.mode != "h_only":
            # Average attention onto the visual feature grid before normalizing.
            weights = F.interpolate(
                heatmap_logits.sigmoid().unsqueeze(1),
                size=self.visual_hw,
                mode="area",
            )[:, 0]
            weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            z_op = (weights.unsqueeze(-1) * v_t.float()).sum(dim=(1, 2))
            z_op = z_op * visibility_logit.sigmoid().unsqueeze(-1)
            result["z_op"] = z_op
        return result
