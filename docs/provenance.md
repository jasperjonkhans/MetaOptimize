# Source provenance and active paths

The original audit is `PROJECT_REVIEW.md` (reviewed outer commit
`61a57f87f0dea05d6514180d21804eae7f460e30`). It describes the pre-remediation
snapshot, not the current behavior. Neither the audit nor these ZIP snapshots
is rewritten by remediation:

| Original file | Bytes | SHA-256 |
| --- | ---: | --- |
| `codes/Supervised_tasks.zip` | 1849452 | `99714221f7ff75fa3a0658313f2b9d0b9815e3d944f7736d261e24530172cdda` |
| `codes/Continual_CIFAR100_experiments.zip` | 2 | `7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6` |

The latter contains only `0D 0A` (CRLF), is not a ZIP, and supplies no
Continual-CIFAR100 implementation. Obtaining the actual original experiment
code is a missing external input; no replacement experiment is invented here.

## Canonical patched sources

Run the extracted, version-controlled `codes/MetaOptimize/` sources, not a new
extraction of the unchanged archive. ZIPs are evidence snapshots and do not
include subsequent fixes. Historical duplicate files remain because external
usage is unknown; their presence does not make them active trainer paths.

The extracted experiments originally started untracked in the outer repository,
and TinyStories contained a nested `.git` repository. For the requested source
commit, that history was preserved locally under the outer repository's
`.git/nested-repositories/`, and TinyStories is tracked as ordinary source and
fixture files, not a Git pointer. The history backup is local Git metadata and
is not included in clones. macOS archive metadata and historical prebuilt
executables are excluded; required tokenizer fixtures remain versioned.
Unchanged ZIP snapshots cannot restore the patches and must not overwrite the
corrected sources.

| Entry point | Active model/data/optimizer path |
| --- | --- |
| `meta_optimize/__init__.py` | public manual PyTorch optimizer in `optimizer.py` |
| `meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py` | separate optional Optax transformation |
| `cifar10/train.py` | local `build_network.py`, `load_data.py`, `Optimizers/build_optimizer.py` |
| `imagenet/train.py` | `torchvision.models` and torchvision datasets; `Optimizers/build_optimizer.py` |
| `tinystories/train.py` | local `model.py`, `tinystories.py`, `build_optimizer.py`, `HF.py` |

ImageNet's historical `build_network.py` (originally identical to CIFAR's)
and `load_data.py` are not the ImageNet trainer's active model/data paths.
TinyStories' `model.configure_optimizers` is not the trainer's optimizer builder.
Do not edit an unused copy and assume the active trainer has changed.

## TinyStories upstream

The inherited code is based on
[karpathy/llama2.c `d9862069e7ef665fe6309e3c17398ded2f121bf5`](https://github.com/karpathy/llama2.c/tree/d9862069e7ef665fe6309e3c17398ded2f121bf5)
(9 October 2023), not today's upstream HEAD. At audit time `export.py`,
`tokenizer.py`, `sample.py`, `test_all.py`, `test.c`, `run.c`, and `runq.c`
matched that upstream snapshot byte-for-byte. `train.py`, `model.py`, and the
data path already contained local modifications. Remediation patches the
active export, tokenization and C safety paths further; source history records
those changes rather than implying they are pristine upstream files.

The old ImageNet README cites `pytorch/examples` but no precise upstream commit
was recovered. The nested TinyStories Git metadata is not a fourth optimizer
implementation and is not required to run the checked sources.

Original `run`/`runq` binaries used a nonportable Compute Canada ELF interpreter
(`/cvmfs/soft.computecanada.ca/gentoo/2020/lib64/ld-linux-x86-64.so.2`). Preserve
legacy artifacts if present, but build current C sources locally into a new
output path; see [validation](validation.md). Binary portability and Windows
execution are not implied by a successful Linux source build.
