# Supervised experiment sources

The active entry points are `cifar10/train.py`, `imagenet/train.py`, and
`tinystories/train.py`. Run each from its containing directory so local imports
resolve, using the installed environment described in the
[root README](../../README.md).

```sh
cd codes/MetaOptimize/cifar10  # from repository root
python train.py --help
```

See [reference commands, data layouts and resume contracts](../../docs/experiments.md),
[historical mathematical variants](../../docs/optimizer-variants.md), and
[source/archive provenance](../../docs/provenance.md).

The `HF` name means Hessian-free, not Hugging Face. Experimental HF AdamW uses
an unnormalized first moment and is not standard `torch.optim.AdamW` at the
same learning rate. Native AdamW baselines are separate implementations.

Install experiment dependencies with `pip install '.[experiments]'` from the
repository root, or use the local `requirements.txt` after selecting matching
Torch/Torchvision builds. W&B and Transformers are optional, not core requirements.
Do not run the archived binaries; rebuild C inference programs locally.
Do not overwrite patched sources with `../Supervised_tasks.zip`.
