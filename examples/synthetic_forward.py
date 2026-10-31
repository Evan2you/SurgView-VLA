"""Compose the released heads using random features, without loading backbones."""

import torch

from surgview_vla import CameraController, RegionGrounder, ViewPredictor


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    batch = 2
    h_t = torch.randn(batch, 4096)
    v_t = torch.randn(batch, 14, 24, 1024)
    depth_t = torch.rand(batch, 45, 80)
    state_t = torch.zeros(batch, 4)

    grounder = RegionGrounder().eval()
    view_head = ViewPredictor().eval()
    # This minimal example omits temporal context and dataset normalization.
    policy = CameraController(input_dim=1024 + 4 + 4, kind="flow").eval()
    with torch.inference_mode():
        operative = grounder(h_t, v_t)
        views = view_head(h_t, operative["z_op"], depth_t)
        condition = torch.cat((operative["z_op"], views["e_view"], state_t), dim=-1)
        actions = policy.predict(condition, steps=32)
        torch.testing.assert_close(views["e_view"], views["g_des"] - views["g_cur"])
        assert actions.shape == (batch, 8, 4)
        assert torch.isfinite(actions).all()

    for name, value in {
        "attention": operative["attention"],
        "z_op": operative["z_op"],
        "e_view": views["e_view"],
        "actions": actions,
    }.items():
        print(f"{name}: {tuple(value.shape)}")
    print("Synthetic forward passed. Random weights; no experimental results or hardware execution.")


if __name__ == "__main__":
    main()
