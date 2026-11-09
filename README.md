# MetaOptimize

Research code for online step-size adaptation. The public Python class is a
specific AdamW-base/Lion-meta implementation, **not** a generic wrapper around
any optimizer. The package, JAX implementation, and experimental HF optimizers
preserve different historical update equations; see the versioned
[optimizer specification](docs/optimizer-variants.md).

## Install and runtime

From the repository root:

```sh
python -m pip install .                 # PyTorch library only
python -m pip install '.[demo]'         # optional MNIST/Torchvision demo
python -m pip install '.[jax]'          # optional JAX/Optax optimizer
python -m pip install '.[experiments,test]' # offline experiment tests
```

Choose matching Torch/Torchvision builds for your platform first. The checked
CPU reference is Python 3.13.5, Torch 2.8.0+cpu, Torchvision 0.23.0+cpu,
NumPy 2.5.4, JAX 0.6.2, Optax 0.2.5 and Chex 0.1.90. These are validation
versions, not a recovered historical paper environment. Metadata permits Python
3.9+ for the core; other versions/platforms and optional dependency combinations
have not all been tested. JAX 0.6.2 requires Python 3.10+.
[Environment setup](docs/validation.md) records the reference install.

The manual core requires Torch >=2.0. Experiments and tests require Torch >=2.3
for `torch.amp.GradScaler(device)`; the test extra requires pytest >=8.2 for
`importorskip(..., exc_type=ModuleNotFoundError)`. These are API-based minimums,
not claims that Torch 2.3 was executed; Torch 2.8 remains the tested reference.

Optional extras: `jax-demo` adds Haiku/TensorFlow for the download-dependent
JAX MNIST demo; `tracking` adds W&B; `export` adds Transformers/Safetensors
for Hugging Face conversion. None is needed for the manual PyTorch core.
The qualified JAX module import also loads the package's Torch export.

## Manual PyTorch usage (no downloads)

```python
import torch
from meta_optimize import MetaOptimize_AdamW_Lion_hg

model = torch.nn.Linear(3, 1)
optimizer = MetaOptimize_AdamW_Lion_hg(
    model.parameters(), alpha0=1e-3, meta_stepsize=1e-3,
    stepsize_blocks="scalar",
)
x, y = torch.ones(4, 3), torch.zeros(4, 1)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
optimizer.step()
```

This is a manual research optimizer, not a `torch.optim.Optimizer` subclass.
Do not assume native scheduler, Trainer, GradScaler, closure, or dynamic
`add_param_group` interoperability. Dense real parameters/gradients are the
supported path; sparse gradients are unsupported. Groups count parameter
**tensors**, not scalar elements: `scalar`, `layerwise`, or an explicit positive
size list. `zero_grad()` zeroes existing gradients rather than setting them to
`None`; absent gradients and zero-valued gradients have different state timing.
`state_dict()` / `load_state_dict()` preserve optimizer/meta state, not model,
RNG or data position. A weight-only load is a warmstart, not optimizer-state resume.
Experiment wrappers have a separate `step(model, loss=None)` protocol; do not
substitute that interface for this class's `step()`.

The optional MNIST example is `python -m meta_optimize.optimizer` and downloads
data; it is not part of offline validation.

## Experiments and tests

Active extracted sources are under `codes/MetaOptimize/{cifar10,imagenet,tinystories}`;
each entry point is **`train.py`**, run from its own experiment directory.
See [experiment commands and reproducibility limits](docs/experiments.md),
[provenance](docs/provenance.md), and [offline checks](docs/validation.md).
For example, inspect a trainer without downloading a dataset:

```sh
(cd codes/MetaOptimize/cifar10 && python train.py --help)
```

The immutable original supervised snapshot is `codes/Supervised_tasks.zip`.
Do not extract it over the patched active sources. The supplied
`codes/Continual_CIFAR100_experiments.zip` contains only CRLF, not an archive:
Continual-CIFAR100 code is missing external input and cannot be reproduced
from this repository. Unit tests and synthetic runs do not reproduce paper
curves or validate full external-dataset/LLM training.

## Citation

```bibtex
@article{sharifnassab2024metaoptimize,
  title={Metaoptimize: A framework for optimizing step sizes and other meta-parameters},
  author={Sharifnassab, Arsalan and Salehkaleybar, Saber and Sutton, Richard},
  journal={International Conference on Machine Learning (ICML)},
  url={https://arxiv.org/pdf/2402.02342},
  year={2025}
}
```
