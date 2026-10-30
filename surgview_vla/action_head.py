from __future__ import annotations

import math
import torch
from torch import nn


class ActionHead(nn.Module):
    """Conditioned MLP or flow-matching head for eight-step camera action chunks."""
    def __init__(self, input_dim, kind='mlp', hidden=256, layers=3, heads=8, horizon=8):
        super().__init__()
        if kind not in ('mlp', 'flow') or hidden % heads or hidden % 2:
            raise ValueError('Invalid head configuration.')
        self.kind, self.horizon, self.hidden = kind, horizon, hidden
        blocks = [nn.Linear(input_dim, hidden), nn.GELU()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.GELU()]
        self.condition = nn.Sequential(*blocks)
        if kind == 'mlp':
            self.output = nn.Linear(hidden, horizon * 4)
        else:
            self.action_in = nn.Linear(4, hidden)
            self.time_in = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.position = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
            layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
            self.output = nn.Linear(hidden, 4)

    def forward(self, x, noisy=None, time=None):
        c = self.condition(x)
        if self.kind == 'mlp':
            return self.output(c).reshape(-1, self.horizon, 4)
        if noisy is None or time is None or noisy.shape != (len(x), self.horizon, 4):
            raise ValueError('Flow requires noisy [B,H,4] and time [B].')
        freq = torch.exp(torch.arange(self.hidden // 2, device=x.device).float()
                         * (-math.log(10000.) / max(self.hidden // 2 - 1, 1)))
        angle = time.float()[:, None] * freq[None] * 1000.
        embedding = self.time_in(torch.cat([angle.sin(), angle.cos()], -1))
        tokens = self.action_in(noisy) + self.position + c[:, None] + embedding[:, None]
        return self.output(self.transformer(tokens))

    @torch.inference_mode()
    def predict(self, x, steps=32):
        if steps <= 0:
            raise ValueError('steps must be positive')
        if self.kind == 'mlp':
            return self(x)
        action = torch.zeros((len(x), self.horizon, 4), device=x.device, dtype=torch.float32)
        for step in range(steps):
            time = torch.full((len(x),), (step + .5) / steps, device=x.device)
            action = action + self(x, action, time).float() / steps
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite flow rollout.')
        return action
