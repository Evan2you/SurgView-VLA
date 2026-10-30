# SurgView-VLA

**Task-Grounded View-to-Action Learning for Autonomous Laparoscopic Camera Control**

[中文说明](README.zh-CN.md)

This repository provides an initial, partial release of selected method components for SurgView-VLA, submitted to IPCAI 2027.

> **Release status:** Core method preview. The complete codebase, training configurations, and reproduction instructions will be released as soon as possible.

## Overview

SurgView-VLA studies task-conditioned camera control through three questions: **what to observe, how the current view should change, and what camera motion to generate**.

The released modules expose this progression at the feature level:

```text
Semantic features + spatial visual features
                  │
                  ▼
       Task-conditioned attention ──► operative feature Z_op
                                              │
Semantic features + relative depth ────────────┤
                                              ▼
                              Current / desired view estimates
                                              │
                                   E_view = g_des − g_cur
                                              │
Z_op + E_view + camera state ──────────────────┤
                                              ▼
                              Conditioned flow action head
                                              │
                                   8 × 4 action chunk
```

This diagram describes the included feature-level example. Backbone feature extraction, temporal-memory integration, and robot execution are outside this release.

## Included components

| Module | What is released |
| --- | --- |
| `What2ViewV0` | Semantic FiLM conditioning of visual features, attention and visibility prediction, visibility-gated attention pooling into `z_op`. |
| `Where2ViewV1` | Independent current and desired view queries, four-dimensional view regression, shared de-normalization, and explicit `g_des - g_cur`. |
| `Where2ViewV11` | An optional frozen-base variant with a gated residual for the desired scale coordinate. |
| `ActionHead` | A conditioned Transformer flow head with zero-initialized, 32-step Euler sampling; an MLP head is also available. |

These are selected implementations extracted from the research code. Their versioned names are retained. Packaging and explanatory comments have been adjusted; the model computations have not been changed. They do not constitute a complete release of every component described in the manuscript.

## Feature and action interfaces

The included view modules use the following default interfaces:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `h_t` | `[B, 4096]` | Task-conditioned semantic feature. |
| `v_t` | `[B, 14, 24, 1024]` | Spatial visual feature grid. |
| `depth_t` | `[B, 45, 80]` | Relative depth map, not metric depth. |
| `attention` | `[B, 45, 80]` | Predicted operative attention. |
| `z_op` | `[B, 1024]` | Attention-pooled, visibility-gated visual feature. |
| `g_cur`, `g_des`, `e_view` | `[B, 4]` | Current view, desired view, and their difference. |
| Flow output | `[B, 8, 4]` | Action chunk ordered as yaw, pitch, roll, insertion. |

The manuscript's geometric descriptor is `[c_x, c_y, log(sqrt(area)), d_rel]`. Both view predictions must be brought into the **same descriptor space** before subtraction. The released head provides shared `view_mean` and `view_scale` buffers; dataset statistics and label construction are not included.

Future expert views provide supervision during training. The released `Where2ViewV1.forward(h_t, z_op, depth_t)` uses current features only: it predicts the desired view and subtracts the predicted current view. It does not require future frames or expert ROI annotations at inference.

`ActionHead` accepts a caller-supplied condition vector. The synthetic example concatenates `z_op`, `e_view`, and a four-dimensional camera state to illustrate the interface. This example does not supply the manuscript's complete temporal context, learned normalization, or perception pipeline.

## Run the synthetic example

Use Python 3.10+ with PyTorch. The modules and checks were exercised on CPU with Python 3.10.12 and PyTorch 2.6.0.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m examples.synthetic_forward
python -m unittest discover -s tests -v
```

The example uses random features and randomly initialized weights. It checks module composition and tensor shapes; it does **not** reproduce experimental results or generate a trained control policy.

## Release scope

Included: the core modules above, a synthetic forward example, and CPU checks for pooling, view subtraction, the optional scale residual, and flow sampling.

Still to be released: the complete training and evaluation pipeline, data preparation and annotation processing, backbone integration, temporal context, model weights, normalization statistics, experiment configurations, and reproduction instructions. Robot calibration and execution integration are also absent from this preview. Action outputs from this package are not ready for execution on hardware.

No experimental scores are claimed for the synthetic example. A license has not yet been specified for this preview.

**The code repository will be completed as soon as possible.**
