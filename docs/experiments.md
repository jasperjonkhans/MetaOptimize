# Experiment commands and reproducibility contracts

Run each trainer from `codes/MetaOptimize/<experiment>` in the environment
specified in [validation](validation.md). The commands below describe the
patched extracted sources, not the unchanged ZIP snapshot. `HF` means
Hessian-free; its historical AdamW rule is **not** native AdamW (see
[specification v1](optimizer-variants.md)). Use a new run name for independent
experiments. Never load untrusted Torch checkpoints.

## Data-free reference smokes

From the repository root, with `.[experiments,test]` installed:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_vision_training.py -k 'cli_and_tiny_cpu_main_without_downloads and cifar10'
python -m pytest -q tests/test_vision_training.py -k 'cli_and_tiny_cpu_main_without_downloads and imagenet'
python -m pytest -q tests/test_vision_training.py -k real_resnet_cpu_synthetic_image_update
python -m pytest -q tests/test_tiny_training.py -k two_step_train_eval_save_resume
```

The vision CLI smokes replace dataset/model construction with a tiny synthetic
fixture; the separate ResNet test uses real ResNets and synthetic images. They
are not CIFAR/ImageNet accuracy measurements. The TinyStories smoke uses real
small Transformers and locally generated token IDs, including train/eval,
saving and continuation. None downloads datasets or models. The full test
files add loss weighting, seeds, budget boundaries, optimizer and resume checks.

For a standalone synthetic TinyStories CLI run, first create throwaway token
shards (no tokenizer/model download):

```sh
python - <<'PY'
from pathlib import Path
import numpy as np
p = Path('/tmp/metaopt-tiny-smoke/data/tok32')
p.mkdir(parents=True, exist_ok=True)
for i in range(3):
    ((np.arange(97, dtype=np.uint16) + i * 7) % 32).tofile(p / f'shard{i}.bin')
PY
(cd codes/MetaOptimize/tinystories && python train.py \
  --device cpu --dtype float32 --vocab-source custom --vocab-size 32 \
  --dim 8 --n-layers 1 --n-heads 2 --n-kv-heads 2 --multiple-of 4 \
  --max-seq-len 4 --batch-size 2 --gradient-accumulation-steps 2 \
  --max-iters 2 --eval-iters 1 --eval-interval 1 \
  --warmup-steps 0 --total-steps 4 --optimizer HF --alpha0 .002 --seed 31 \
  --data-dir /tmp/metaopt-tiny-smoke/data \
  --save-directory /tmp/metaopt-tiny-smoke/outputs --run-name smoke)
```

Use a fresh `/tmp/metaopt-tiny-smoke` directory on repeat runs. Missing tokenizer
metadata is explicitly recorded for synthetic pretokenized inputs; these IDs
are not a usable text model/tokenizer pair.

## CIFAR-10 (external data; not run as validation)

```sh
cd codes/MetaOptimize/cifar10
python train.py --dataset CIFAR10 --NN-name ResNet18 --optimizer HF \
  --seed 1 --num-epochs 1 --batch-size 100 --max-time 00:05:00 \
  --workers 0 --deterministic --save-directory outputs --run-name cifar-reference
```

The active loader uses `./data` relative to this directory and torchvision's
CIFAR-10 layout (`data/cifar-10-batches-py/`). It requests download if data is
absent; therefore this is **not** an offline smoke. `--dataset MNIST` with a
compatible `--NN-name M1` or `M2` uses `./data/MNIST`. Check `--help` and the
model builder for supported architectures; no `--data-dir` is offered here.

## ImageNet (external data; not run as validation)

Supply an ImageFolder tree with the **literal** split name `validation`, not
`val`:

```text
/path/to/imagenet/train/<class>/*
/path/to/imagenet/validation/<class>/*
```

```sh
cd codes/MetaOptimize/imagenet
python train.py /path/to/imagenet --arch resnet18 --optimizer HF \
  --seed 1 --epochs 1 --batch-size 32 --workers 0 --max-time 00:05:00 \
  --deterministic --save-directory outputs --run-name imagenet-reference
```

Do not add `--pretrained` for offline work: that option can download weights.
`--dummy` uses local torchvision FakeData, but its large default sample count
is a benchmark path, not the tiny regression fixture. Distributed validation
uses padded equal-length shards and can count duplicate samples; the manifest
records that limitation. CPU/Gloo regression coverage does not establish GPU,
NCCL or multi-node behavior.

## TinyStories data preparation and export

For real data, place JSON arrays of objects with a `story` field in
`DATA/TinyStories_all_data/*.json`. The optional `download` stage accesses a
moving upstream URL and extracts external data; it is not offline validation.
A custom SentencePiece workflow from `codes/MetaOptimize/tinystories` is:

```sh
python tinystories.py train_vocab --vocab_size 2048 --data-dir /path/to/DATA
python tinystories.py pretokenize --vocab_size 2048 --data-dir /path/to/DATA \
  --tokenizer-model /path/to/DATA/tok2048.model --max-workers 2
python train.py --device cpu --vocab-source custom --vocab-size 2048 \
  --tokenizer-model /path/to/DATA/tok2048.model --data-dir /path/to/DATA \
  --max-iters 2 --eval-iters 1 --eval-interval 1 \
  --warmup-steps 0 --batch-size 2 --gradient-accumulation-steps 1 \
  --save-directory outputs --run-name tiny-reference
```

Vocabulary training expects enough local text; the recipe is not a promise
that an arbitrary short corpus supports a requested vocabulary. With the
original Llama tokenizer, use `pretokenize --vocab_size 0` and train with
`--vocab-source llama2 --vocab-size 32000` plus the matching tokenizer.
Token shards are raw `uint16`: custom shards live in `DATA/tok<N>/*.bin`,
Llama-tokenizer shards in `DATA/TinyStories_all_data/*.bin`. The first sorted
shard is validation, the rest training; at least two nonempty sufficiently
long shards are required. Worker failures propagate and successful
preprocessing writes a shard/tokenizer hash manifest. Keep the tokenizer and
manifest with the shards; token IDs must fit the selected vocabulary.

Training writes `outputs/<run-name>/config.json`, `ckpt.pt` and version-0
`model.bin`, with TensorBoard/W&B only when requested. The final checkpoint is
saved even without a validation improvement. Explicit export commands:

```sh
python export.py /tmp/metaopt-model.bin --checkpoint outputs/tiny-reference/ckpt.pt --version 0
python export.py /tmp/metaopt-hf --checkpoint outputs/tiny-reference/ckpt.pt --version -1 --dtype fp16
```

The latter needs the optional `export` dependencies; a complete Transformers
roundtrip is outside core validation. Binary versions 0–2 are fixed-format,
not general dtype selection. Locally compile C inference/tokenizer programs
using [these commands](validation.md#c-build-without-overwriting-historical-binaries).
For meaningful inference pair a compatible exported model with the exact
matching tokenizer, not the synthetic-ID fixture above.

## Seeds, budgets, schedules and manifests

- Vision seeds Python/NumPy/Torch, loader generators and workers;
  `--deterministic` requests deterministic algorithms (may cost performance or
  reject unsupported kernels). TinyStories seeds model/data by CLI seed,
  rank and worker. Equal seeds do not promise equality across devices,
  dependency versions or changed process/worker counts.
- `--max-time hh:mm:ss` is a **soft invocation budget**, checked between
  batches/updates, not a preemptive timeout for in-flight compute, data I/O,
  collectives, export or checkpoint writes. Setup counts toward the budget;
  checkpoint finalization may finish after it. Vision also stops at its epoch
  bound; TinyStories caps attempted accumulated iterations at `--max-iters`,
  including attempts whose optimizer update is skipped by AMP.
- TinyStories accumulates scaled microbatch losses, unscales/clips once, then
  attempts one optimizer update. `iter_num` counts completed accumulation attempts;
  `optimizer_updates` counts only executed optimizer steps. Logs expose
  `optimizer_step_skipped` per attempt and cumulative `optimizer_step_skips`;
  results also return the cumulative skip count. Skipped attempts still consume
  data and the attempt budget; persistent overflow does not trigger endless retries.
  Global batch size is per-rank batch size times accumulation steps times world
  size. Nominal tokens per iteration describe attempts, not successful updates;
  report consumed microbatches/tokens, actual updates and measured duration.
- Native baseline scheduler bounds are effective settings, not the old hidden
  10000/422000 constants. Vision derives a horizon from epochs and loader
  length unless `--total-steps` is supplied; TinyStories defaults it to
  `--max-iters`. `--warmup-steps` controls warmup. Native AdamW and HF use
  `--alpha0`; ImageNet SGD/SGDm/Adam use `--base-lr` (and `--momentum` where
  applicable). CIFAR-10 and TinyStories offer only HF/AdamW optimizer choices.
- Vision stores `manifest.json`, metric history, checkpoints and TensorBoard
  under `<save-directory>/<run-name>/`; existing run directories are rejected.
  The manifest records effective arguments, optimizer/scheduler, architecture,
  precision/device and dataset identity. ImageFolder's path/label fingerprint
  is **not** an image-content hash. TinyStories `config.json` records effective
  model/optimizer settings, precision, actual compile/attention mode, batch
  semantics and full shard hashes (startup hashing may be expensive).
  A missing tokenizer file/hash is recorded, not a provenance guarantee.
- TinyStories `--compile` actually requests compilation; `--attention manual`
  versus `sdpa` selects the implementation. SDPA chooses its backend kernel;
  selecting it does not prove a particular FlashAttention kernel ran.

## Resume is not a weight warmstart

**Manual package:** optimizer `state_dict()` preserves its base/meta state and
configuration. The caller must separately save model, RNG and data position;
it is not a complete trainer checkpoint.

**Vision:** use `--resume PATH` with a **new** run name for supported
single-process completed-epoch checkpoints. The checkpoint includes model,
optimizer/meta/scheduler state, RNG and loader-generator state, metrics and
configuration; incompatible resumes are rejected. Interrupted epochs and
DDP checkpoints are not exact replay. Use explicit `--warmstart PATH` to load
only weights from those or legacy checkpoints. Do not claim mid-epoch or
cross-world-size continuation.

**TinyStories:** `--init-from resume --checkpoint PATH` restores model,
optimizer/meta/scheduler, scaler, RNG, attempted iteration, successful
`optimizer_updates` count and consumed-data position from a version-2 checkpoint.
Version-1 and older checkpoints lack trustworthy update accounting and require
explicit warmstart; their successful-update counts are not inferred.
Exact replay requires **one process and `--num-workers 0`**, matching effective
configuration and shard identity. The stream replays every previously consumed
microbatch (O(consumed batches) I/O), rather than caching the dataset.
Uninterrupted-run equivalence applies at completed-accumulation checkpoint
boundaries. Budget-interrupted partial accumulations discard gradients but retain
consumed-data position and current RNG state; resume continues from the saved
model/RNG/data state without reconstructing the uncommitted partial attempt.
Preserve `--total-steps` when extending `--max-iters`, otherwise the default horizon
changes and the compatibility check correctly rejects the resume.
`--init-from warmstart --checkpoint PATH` loads only model weights into a fresh
training history; legacy/DDP/multi-worker checkpoints require this explicit
mode. Reusing the original run directory for an explicit TinyStories resume
updates its latest checkpoint/config; retain a copy or use a new run name if
that history must remain immutable.

Full dataset training, paper curves, pretrained LLM quality, comprehensive GPU
behavior and the absent Continual-CIFAR100 input remain outside validation.
