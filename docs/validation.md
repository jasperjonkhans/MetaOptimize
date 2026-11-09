# Offline validation and runtime

## Reference CPU environment

The remediation environment uses Python **3.13.5**, Torch **2.8.0+cpu**,
Torchvision **0.23.0+cpu**, NumPy **2.5.4**, pytest **9.2.0**, JAX/jaxlib
**0.6.2**, Optax **0.2.5**, Chex **0.1.90**, TensorBoard **2.21.0**, and
SentencePiece **0.2.3**. The complete observed environment is recorded in
[validation-runtime.txt](validation-runtime.txt). It is a validation snapshot,
not a lockfile for every platform or a recovered paper environment. GPU wheels,
Haiku, TensorFlow, Transformers and W&B are not required for core tests.

One way to create the reference runtime (network is needed for installation,
not for the synthetic test runs):

```sh
uv venv --python 3.13 /tmp/metaopt-venv
uv pip install --python /tmp/metaopt-venv/bin/python \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/metaopt-venv/bin/python \
  '.[experiments,test,jax]' jax==0.6.2 optax==0.2.5 chex==0.1.90 build
. /tmp/metaopt-venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
```

For the actual remediation run, the shared interpreter lives **outside** the
repository at `/tmp/metaoptimize-review-2s5ec4ay/venv/bin/python`. No existing
project environment was modified. Use the freeze if exact dependency versions
matter, with the CPU wheel index available for Torch's `+cpu` builds.

## Hermetic tests

From the repository root, with the relevant extras installed, prefer two
separate Python processes for combined-stack validation:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --ignore=tests/test_jax_optimizer.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_optimizer.py
python -m build
```

This runs the non-JAX cases first and the real JAX/JIT cases separately, without
skipping supported cases or suppressing warnings. Forking preprocessing or
DataLoader workers after JAX initializes its multithreaded runtime can warn
about deadlocks. `--ignore` prevents collection from importing the JAX module;
`-m 'not jax'` does not. Full discovery remains available with
`python -m pytest -q`, but the split commands above are preferred when JAX is
installed and are used by CI.

`testpaths = ["tests"]` intentionally excludes the historical
`codes/MetaOptimize/tinystories/test_all.py`: those inference tests can download
model/tokenizer fixtures and are not offline optimizer tests. Keep them as
separate optional integration checks, not evidence of optimizer correctness.
The optional JAX suite skips when its dependencies are unavailable; a skipped
suite is **not** JAX validation. No MNIST/CIFAR/ImageNet datasets or pretrained
language models should be downloaded for the synthetic suite.

The CI workflow installs CPU dependencies, runs the synthetic suite in the two
processes above, builds packaging, and compiles the five preserved C tokenizer fixtures. CI installation
requires package-network access; test execution uses only local synthetic data.

## C build without overwriting historical binaries

From the repository root, with GCC and libc/libm development tools:

```sh
gcc -O2 -o /tmp/metaopt-run codes/MetaOptimize/tinystories/run.c -lm
gcc -O2 -o /tmp/metaopt-runq codes/MetaOptimize/tinystories/runq.c -lm
gcc -O2 -o /tmp/metaopt-tokenizer-test codes/MetaOptimize/tinystories/test.c -lm
(cd codes/MetaOptimize/tinystories && /tmp/metaopt-tokenizer-test)
```

The working directory in the last command is required for the existing
`tokenizer.bin`/`test/tokenizer_*.bin` fixtures. It should print `ALL OK`.
Do not execute the old checked-in `run`/`runq` binaries as a portability check.
Source compilation and tokenizer fixtures do not establish C/Python model
numerical equivalence or general parser/kernel safety.

## What this does not validate

These checks do not reproduce full MNIST, CIFAR-10, ImageNet, TinyStories or
paper runs, pretrained model quality, or the missing Continual-CIFAR100 code.
No GPU/AMP/NCCL/multi-node, Windows, complete Transformers export/import, or
historical Torch 2.0.1 stack is implied. Read the final remediation results for
which individual checks were actually executed; commands here are reference
instructions, not a claim that every optional external workflow was run.
