# InMoE

Official code release for **InMoE: Interaction-Aware Graph Mixture of Experts for Trajectory Prediction**.

This repository is kept intentionally lightweight. The released source code lives under `src/`, together with a small set of entry scripts used to call the main classes.

## Layout

```text
InMoE/
├── README.md
└── src/
    └── source code and entry scripts
```

## Example

One minimal class call example:

```python
# run inside one source directory, for example: src/InMoE-Argoverse/
from model import InMoE

model = InMoE(
    hidden_dim=128,
    num_historical_steps=20,
    num_future_steps=30,
    pos_duration=20,
    pred_duration=20,
    a2a_radius=50.0,
    l2a_radius=50.0,
    num_visible_steps=2,
    num_modes=6,
    num_attn_layers=2,
    num_hops=4,
    num_heads=8,
    dropout=0.1,
    lr=3.5e-4,
    weight_decay=1e-4,
    warmup_epochs=4,
    T_max=72,
    is_training=True,
)
```

The exact runtime arguments depend on the entry script or experiment setup you use locally. Dataset files and full experiment configuration are intentionally not included in this repository.

## Citation

If you find this code useful, please cite the corresponding paper.

```bibtex
@article{inmoe2026,
  title   = {InMoE: Interaction-Aware Graph Mixture of Experts for Trajectory Prediction},
  author  = {Anonymous},
  journal = {Pattern Recognition},
  year    = {2026}
}
```
