# MetaOptimize project-review remediation results

Date: 8 November 2026. Input: `PROJECT_REVIEW.md`, reviewing repository commit `61a57f87f0dea05d6514180d21804eae7f460e30` and the extracted experiments.

## Summary

All 20 numbered findings have been addressed through fixes, explicit supported contracts, mathematical specifications and regression tests. This does not mean that missing historical inputs or unexecuted large-scale experiments have been recovered or validated.

- B01–B12: confirmed implementation defects repaired.
- M01–M02: historical mathematical variants preserved, named, documented and tested, as explicitly requested. They were not silently standardized or declared paper-equivalent.
- R01–R03: reproducible future-run configuration, seeds, budgets, artifacts and supported resume paths implemented. Unsupported exact-resume modes are explicitly rejected rather than silently warmstarting.
- W01–W02 and T01: installation/API documentation, dependency separation, provenance, offline test discovery and CI added or corrected.
- One external-input blocker remains: the supplied Continual-CIFAR100 archive is only two CRLF bytes. It contains no implementation to repair or reproduce.

Seven isolated implementation workers covered the package, JAX, experimental optimizers, vision training, TinyStories training/data, export/C safety, and documentation/validation. Changes were integrated and tested together before application to the original checkout. The delivered change covers 40 files: 17 implementation files, 11 test modules, 11 documentation/configuration files and this report. The actual delivered checkout was then tested again. Independent reviews also found checkpoint-loader and AMP-accounting defects; these were reproduced and repaired before delivery.

Independent integration review is complete. The reported loader, clock and AMP-accounting findings are closed; the final bounded review accepted the actual-call accounting fix after independently reproducing both 170-attempt saturation cases. This is scoped review evidence, not a guarantee that all possible defects are absent.

## Results by finding

| Finding | Result | Implementation and evidence |
|---|---|---|
| B01 — accumulation/clipping | Fixed | TinyStories backpropagates every scaled microbatch, synchronizes at the accumulation boundary, unscales before clipping, and applies one update using existing gradients. HF and native AdamW match concatenated batches, including real CPU GradScaler and skipped-overflow cases. See `tests/test_tiny_training.py`. |
| B02 — missing-gradient blocks | Fixed | Package membership follows original parameters, not the active-gradient subset. Partially active blocks finish their meta update; completely inactive blocks freeze. First/middle/last missing gradients and the distinction between None and zero are tested in `tests/test_package_optimizer.py`. |
| B03 — JAX tree inner product | Fixed | Corresponding leaves are multiplied with `vdot` before scalar reduction. Scalar/vector/matrix/nested-tree and zero-trace cases pass real eager/JIT tests in `tests/test_jax_optimizer.py`. |
| B04 — package validation | Fixed | Constructor rejects empty/duplicate/invalid parameters, malformed blocks and nonfinite or out-of-range hyperparameters before state construction. Sparse updates fail before partial mutation. Zero meta step size and gamma=1 remain supported. |
| B05 — ImageNet optimizer interfaces | Fixed | Every offered ImageNet optimizer works through the common gradient/state protocol. Canonical parser arguments and historical namespace aliases are resolved consistently; actual rates, momentum and decay are tested against real parsers. CIFAR and TinyStories advertise only their supported HF/AdamW choices. |
| B06 — experimental grouping | Fixed | Float conversion and stable group initialization are corrected. Scalar, layerwise and valid explicit/historical partitions work; unsupported names and invalid sizes fail early with diagnostic errors. Historical tensor-count layouts are preserved, not guessed from model names. |
| B07 — HF/DDP | Fixed | HF consumes backward-generated synchronized `.grad` rather than bypassing DDP with `autograd.grad`. Real two-process CPU/Gloo tests compare rank parameters and every optimizer/meta state with a global-batch reference, including accumulated TinyStories updates and the ImageNet entrypoint. |
| B08 — JAX execution barriers | Fixed | Supported tree APIs, a JIT closure over the optimizer, tree/callable masks and explicit required-parameter checks are implemented. Optional demo dependencies are imported only by the demo; core eager/JIT updates execute with real JAX. |
| B09 — metrics/logging | Fixed | CIFAR loss is weighted by actual sample count; ImageNet reports epoch-average loss; TinyStories logs aggregate microbatch loss and actual optimizer rates. Recorder tests exercise TensorBoard/W&B-facing values without online services. |
| B10 — exports | Fixed | Legacy export no longer mutates vocabulary configuration; repeated exports are identical. HF dtype dispatch and CLI conversion are corrected. Invalid shapes/options produce useful errors. Real small-model exports and dtype-dispatch tests pass; a complete Transformers roundtrip was not run. |
| B11 — tokenization failures | Fixed | Process-pool results are consumed, errors propagate, expected shard names/sizes/hashes are verified, and a success manifest is committed only for a complete result. Real local SentencePiece preprocessing, corrupt workers, stale manifests and tampering are tested. |
| B12 — C prompt overflow | Fixed | Both C implementations validate user/system lengths, reject oversized interactive lines, handle EOF, bound rendering and size token storage. Both accept 0/511-byte inputs and reject 512/600-byte inputs under actual ASan/UBSan checks; original tokenizer fixtures remain intact. |
| M01 — experimental AdamW convention | Specified and preserved | Experimental unnormalized first moments and second-moment correction remain historical behavior, explicitly distinct from native AdamW. Hand calculations pin `1 -> .9 -> .71`, changing gradients and decay. See `docs/optimizer-variants.md`. |
| M02 — ordering/trace variants | Specified and tested | Package/JAX/experimental Lion ordering, alpha timing, epsilon and trace conventions remain separately specified. The definite JAX mask inconsistency was repaired so weight and trace decay use the same effective mask. No cross-framework equivalence is claimed. |
| R01 — seeds/budgets | Addressed | CLI seeds reach model, data and workers; JAX shuffling uses a seeded generator. Monotonic soft budgets are checked inside training, without the old hidden epoch reservation. TinyStories caps accumulation attempts and separately records executed optimizer updates and AMP skips. Fake clocks, rank/worker streams and deterministic CPU cases are tested. |
| R02 — artifacts/resume | Addressed within explicit limits | Outputs live under run-specific directories and accidental scratch-run overwrites are rejected. Package/HF/native optimizers serialize their state; trainers include model, optimizer/scheduler, RNG and applicable scaler/data position. Supported exact continuations match uninterrupted runs; unsupported modes require explicit weight-only warmstarts. See limits below. |
| R03 — effective configuration | Addressed for new runs | Effective manifests record actual optimizer/scheduler settings, batch/token semantics, precision/device/execution modes, seeds and dataset/tokenizer identity. Compile/attention controls reflect actual behavior; hidden schedule constants are removed. Historical paper-to-run configuration cannot be reconstructed from absent original manifests. |
| W01 — API/install/docs | Addressed | README paths/commands and public manual-optimizer limitations are corrected. Demo/JAX/experiment/tracking/export/test dependencies are separated. Trainers are import-safe, TensorBoard requirements are declared, and final packaging is checked outside repository source imports. |
| W02 — drift/provenance | Addressed | Active versus historical copies, upstream llama2.c commit, local deltas and unchanged archives are documented. Duplicate fixes share a regression matrix; different mathematical variants remain separate. Unproven-unused legacy files and nested history were not deleted. |
| T01 — research-code tests | Addressed | Eleven hermetic test modules cover optimizer equations, blocks, accumulation, AMP, DDP, CLI/builders, budgets, metrics, preprocessing, resume and C/export boundaries. Default pytest discovery excludes historical download-dependent inference tests; optional dependency skips are explicit. |

## Validation actually executed

Reference environment: Python 3.13.5; Torch 2.8.0+cpu; Torchvision 0.23.0+cpu; JAX/jaxlib 0.6.2; Optax 0.2.5; Chex 0.1.90; NumPy 2.5.4; pytest 9.2.0. The full installed reference set is recorded in `docs/validation-runtime.txt`. This is not a reconstruction of the historical paper environment.

| Gate | Result |
|---|---|
| Delivered-checkout suite, separate CPU/JAX processes | **503 passed, 17 skipped, 133 subtests passed**: 468 non-JAX cases in 41.27 seconds, followed by 35 JAX cases in 4.54 seconds. |
| Package regressions | 19 passed; one CUDA-only load-migration case skipped; 70 subtests passed. Includes real Torch AdamW comparison and exact serialized continuation. |
| JAX regressions | 35 passed using real CPU eager/JIT execution, not framework stubs. |
| Experiment optimizer/parser/state/DDP regressions | 373 passed; 16 parametrizations skipped because those optimizers are not offered by that experiment. Includes real Gloo processes, CPU GradScaler and checkpoint-rejection atomicity. |
| Vision regressions | 36 passed, including actual ResNet synthetic-image updates, real parser/builder entrypoints, worker replay and two-rank Gloo training. |
| TinyStories regressions | 25 passed, including real small Transformers, FP32/BF16, CPU FP16 overflow accounting, accumulation, tokenization, DDP and exact supported replay. |
| Export/C safety regressions | 15 passed, with original tokenizer fixtures and real Clang ASan+UBSan execution for both C implementations. |
| Genuine core + test-only installation | **423 passed, 40 skipped, 133 subtests passed**. The environment contains Torch, NumPy and pytest but no experiment/JAX extras; missing optional modules skip clearly without fixture errors or failed subprocesses. |
| Standalone documented TinyStories CLI | Real two-step CPU run completed without downloads and wrote `ckpt.pt`, `model.bin` and `config.json`. This is separate from monkeypatched dataset/model entrypoint tests. |
| Final integrated wheel/source build | Passed; wheel contains only the three intended package modules. |
| Fresh installed-wheel core smoke | Passed from `/tmp` in a genuinely fresh Torch-only environment: import, real update, versioned state roundtrip/continuation and invalid-constructor rejection. NumPy/JAX/Torchvision/TensorBoard/SentencePiece/Haiku/TensorFlow/Transformers were absent. |
| Whole-source syntax/focused lint | All 35 Python files parsed; Ruff `F821,F822,F823,E9` and whitespace checks passed. |

The 17 full-environment skips are exactly 16 deliberately unoffered optimizer combinations and one unavailable-CUDA migration case. They are not silently missing JAX or CPU/DDP coverage. An earlier combined 451-case invocation emitted 47 multiprocessing warnings after threaded-runtime initialization, although it completed. CI and the preferred final gate therefore run JAX in a separate process; the final split gate completed without those warnings. This is not a general threaded/fork safety proof.

The validation interpreter remains available for immediate reruns at `/tmp/metaoptimize-review-2s5ec4ay/venv/bin/python`; it is temporary, so recreate dependencies from the documentation after system cleanup.

After installing the documented extras, run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --ignore=tests/test_jax_optimizer.py`, then `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_optimizer.py`. See `docs/validation.md` for installation, optional suites, C builds and packaging, and `docs/experiments.md` for per-dataset synthetic/reference commands. The GitHub Actions workflow was added but was not itself executed on GitHub during this remediation.

## Independent-review follow-up

Independent reviews found and closed three categories of additional defect:

1. HF accepted nonfinite/negative-second-moment state, sparse tensors and out-of-domain bias-correction products. Loading now stages and validates dense floating tensors, destination conversion, finite values, algorithm-specific second moments, derived rates and clock products in [0, 1] before committing anything. Rejected loads preserve model/live/input state and valid continuation.
2. Native rollback replaced parameter groups while the wrapper retained an obsolete alias. The wrapper now exposes the underlying optimizer's live groups through a property, including after rejected loads; learning-rate reporting and GradScaler continue to use actual groups.
3. TinyStories counted AMP-skipped attempts without an actual-update counter. It now logs/persists separate attempts, successful updates and skips. For each of HF/AdamW, a real CPU FP16 caller-level test records two overflow attempts, zero updates, two skips and four consumed microbatches, then two finite resumes correctly increase successful updates to one and two. A further 170-attempt permanent-overflow case for each optimizer drives GradScaler to scale zero and still records zero actual updates, 170 skips and 340 consumed microbatches. Counting observes actual optimizer calls, not scale-change heuristics.

All were reproduced against pre-fix committed sources. The final integrated suite includes rejection/continuation and actual training-loop regressions. TinyStories checkpoint format is now v2; exact resume requires a valid successful-update count, while older or incomplete checkpoints require explicit weight warmstart.

## Supported limits and remaining blockers

- **Missing external code:** `codes/Continual_CIFAR100_experiments.zip` is exactly CRLF, not a ZIP. A valid archive or upstream source is required. No replacement implementation was invented.
- **Exact resume:** vision supports single-process completed-epoch boundaries only. TinyStories supports one process and `num_workers=0`, replaying O(consumed microbatches) I/O. Uninterrupted-run equivalence applies at completed accumulation boundaries: a budget-interrupted partial accumulation discards gradients while preserving consumed-data/RNG position, and those uncommitted gradients are not reconstructed. Partial-epoch vision and distributed/multiworker/legacy states are explicitly warmstart-only where applicable. Preserve effective schedule bounds when extending a run.
- **Soft budgets:** checks do not preempt setup, in-flight compute/data loading/collectives, hashing, exports or checkpoint writes. Such work can finish after the nominal time limit.
- **Dataset identity:** TinyStories hashes shard contents; ImageFolder fingerprints cover indexed paths/labels, not every image byte. Legacy pretokenized data can lack tokenizer provenance and records that absence. Optional download stages still depend on external upstream availability; retain verified local inputs/manifests.
- **Distributed metrics:** padded distributed samplers can count repeated samples. This limitation is recorded rather than presented as an exact unpadded validation metric.
- **Not executed:** full MNIST/CIFAR/ImageNet/TinyStories training, historical paper curves, GPU/CUDA/NCCL/multi-node or Windows validation, real online W&B, the actual torch.compile backend, pretrained-model inference, legacy download-dependent Python inference tests, and a complete Transformers roundtrip. CPU BF16/GradScaler/Gloo/C sanitizer tests are real but do not prove these other paths.
- **Security boundary:** repaired C chat bounds are not a comprehensive audit of untrusted checkpoints, tokenizer/model formats, downloads or all C kernels. Load only trusted Torch checkpoints.
- **Source delivery:** patched extracted sources are canonical; unchanged ZIP snapshots still contain historical defects. The experiments originally started untracked. At the user-requested commit follow-up, TinyStories was included as ordinary source/fixture files rather than an unusable Git pointer. Its nested Git history was preserved locally under the outer `.git/nested-repositories/`; macOS metadata and historical prebuilt executables are excluded. Do not overwrite corrected sources by re-extracting the ZIP.

## Original-input integrity

`PROJECT_REVIEW.md` and both supplied archives remain unchanged:

- Review SHA-256: `aae1e981aa2d10a03c79e48514ece45eb7e1e29dc98ed37dc4566285135f1fcf`.
- `codes/Supervised_tasks.zip`: 1,849,452 bytes; SHA-256 `99714221f7ff75fa3a0658313f2b9d0b9815e3d944f7736d261e24530172cdda`.
- `codes/Continual_CIFAR100_experiments.zip`: 2 bytes; SHA-256 `7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6`.

Existing binary/tokenizer fixtures and historical metadata were preserved. At the original remediation delivery, no commit or staging operation had been performed in the user's checkout; isolated worktree commits were used for handoff and integration. The subsequent user-requested commit includes the canonical experiment sources and required fixtures; nested Git history remains preserved locally and is not part of the committed tree.
