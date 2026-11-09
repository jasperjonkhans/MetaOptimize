"""Offline real-PyTorch regressions for tinystories/train.py and tinystories.py."""
import copy
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


# These experiment imports are required by tinystories.py, even for offline tests.
for dependency in ("requests", "tqdm", "sentencepiece"):
    pytest.importorskip(dependency, exc_type=ModuleNotFoundError)


SOURCE = Path(__file__).resolve().parents[1] / "codes/MetaOptimize/tinystories"


@pytest.fixture(scope="module")
def lane():
    # These historical scripts use sibling imports, not a Python package. Isolate
    # their generic module names from other experiment lanes during collection.
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(SOURCE))
    for name in ("train", "model", "tinystories", "tokenizer", "export", "build_optimizer", "HF"):
        patch.delitem(sys.modules, name, raising=False)
    train = importlib.import_module("train")
    data = importlib.import_module("tinystories")
    builder = importlib.import_module("build_optimizer")
    yield train, data, builder
    patch.undo()


def tiny_args(train, tmp_path, name="run", optimizer="AdamW", *extra):
    return train.parse_args([
        "--device", "cpu", "--vocab-source", "custom", "--vocab-size", "32",
        "--dim", "8", "--n-layers", "1", "--n-heads", "2", "--n-kv-heads", "2",
        "--multiple-of", "4", "--max-seq-len", "4", "--batch-size", "2",
        "--gradient-accumulation-steps", "2", "--max-iters", "2", "--eval-iters", "1",
        "--eval-interval", "1", "--warmup-steps", "0", "--total-steps", "4",
        "--optimizer", optimizer, "--alpha0", ".002", "--seed", "31",
        "--save-directory", str(tmp_path / "outputs"), "--run-name", name,
        "--data-dir", str(tmp_path / "data"), *extra,
    ])


def tiny_data(tmp_path):
    directory = tmp_path / "data" / "tok32"
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        ((np.arange(97, dtype=np.uint16) + i * 7) % 32).tofile(directory / f"shard{i}.bin")


def transformer(train, dropout=0):
    return train.Transformer(train.ModelArgs(dim=8, n_layers=1, n_heads=2, n_kv_heads=2,
                                             vocab_size=32, multiple_of=4, max_seq_len=4,
                                             dropout=dropout))


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_tree_equal(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("optimizer_name", ["HF", "AdamW"])
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "cpu-fp16-scaler"])
def test_accumulated_clipped_update_matches_concatenated(lane, tmp_path, optimizer_name, amp):
    train, _, builder = lane
    train.seed_everything(12)
    accumulated = transformer(train)
    concatenated = copy.deepcopy(accumulated)
    args = tiny_args(train, tmp_path, "unused", optimizer_name)
    actual = builder.build_optimizer(accumulated, args, None)
    expected = builder.build_optimizer(concatenated, args, None)
    x = torch.randint(0, 32, (4, 4))
    y = torch.randint(0, 32, (4, 4))
    scaler = torch.amp.GradScaler("cpu", enabled=amp, init_scale=128)
    expected_scaler = torch.amp.GradScaler("cpu", enabled=amp, init_scale=128)
    # A CPU GradScaler scales real fp32 forward/backward gradients here; CPU
    # autocast is separately exercised in test_two_step_train_eval_save_resume.
    for _ in range(2):
        losses = []
        for xx, yy in zip(x.chunk(2), y.chunk(2)):
            accumulated(xx, yy)
            losses.append(accumulated.last_loss.item())
        returned = train.accumulated_step(accumulated, accumulated, actual,
                    iter(zip(x.chunk(2), y.chunk(2))), 2, scaler, .01)
        assert returned == pytest.approx(sum(losses) / 2, rel=1e-6)
        expected.zero_grad(set_to_none=True)
        concatenated(x, y)
        expected_scaler.scale(concatenated.last_loss).backward()
        expected_scaler.unscale_(expected)
        norm = torch.nn.utils.clip_grad_norm_(concatenated.parameters(), .01)
        assert norm > .01  # The regression must actually exercise clipping.
        expected_scaler.step(expected, concatenated, loss=None)
        expected_scaler.update()
        for a, b in zip(accumulated.parameters(), concatenated.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
    assert scaler.get_scale() == expected_scaler.get_scale()


@pytest.mark.parametrize("optimizer_name", ["HF", "AdamW"])
def test_scaler_skips_nonfinite_update(lane, tmp_path, optimizer_name):
    train, _, builder = lane
    model = transformer(train)
    optimizer = builder.build_optimizer(model, tiny_args(train, tmp_path, "unused", optimizer_name), None)
    before = copy.deepcopy(optimizer.state_dict())
    weights = copy.deepcopy(model.state_dict())
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    optimizer.zero_grad(set_to_none=True)
    loss = next(model.parameters()).sum() * float("inf")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer, model, loss=None)
    scaler.update()
    assert scaler.get_scale() == 64
    assert_tree_equal(optimizer.state_dict(), before)
    assert_tree_equal(model.state_dict(), weights)


class Recorder:
    def __init__(self):
        self.events = []

    def log(self, metrics, step):
        self.events.append((copy.deepcopy(metrics), step))


@pytest.mark.parametrize("optimizer_name", ["HF", "AdamW"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_two_step_train_eval_save_resume(lane, tmp_path, optimizer_name, dtype):
    train, _, _ = lane
    tiny_data(tmp_path)
    full_args = tiny_args(train, tmp_path, "full", optimizer_name, "--dropout", ".15", "--dtype", dtype)
    recorder = Recorder()
    full_result = train.run(full_args, recorder=recorder)
    assert full_result["iter_num"] == 2  # max_iters caps attempts, not an inclusive index.
    assert full_result["optimizer_updates"] == 2 and full_result["optimizer_step_skips"] == 0
    full_dir = tmp_path / "outputs" / "full"
    full = torch.load(full_dir / "ckpt.pt", weights_only=False)
    assert full["iter_num"] == 2 and full["data_batches"] == 4
    assert full["format_version"] == 2 and full["optimizer_updates"] == 2
    manifest = json.loads((full_dir / "config.json").read_text())
    assert manifest["optimizer"] and manifest["global_batch_size"] == 4
    assert manifest["tokens_per_iteration"] == 16 and manifest["precision"] == dtype
    assert manifest["dataset"]["shards"][0]["sha256"]
    assert manifest["dataset"]["tokenizer_status"].startswith("unavailable")
    assert (full_dir / "model.bin").stat().st_size > 0
    assert len(recorder.events) == 4  # evaluation and update each iteration
    assert all(isinstance(m["lr"], float) and m["lr"] == m["learning_rates"][0]
               for m, _ in recorder.events)
    assert [step for m, step in recorder.events if "learning_rates" in m and "loss/val" not in m] == [1, 2]

    # A monotonic fake clock lets one whole iteration complete, then expires.
    # The recorder marks a completed update without relying on CPU wall time.
    partial_args = tiny_args(train, tmp_path, "partial", optimizer_name, "--dropout", ".15", "--dtype", dtype)
    class StopAfterFirstUpdate:
        def __call__(self):
            return 86401 if recorder_partial.events and recorder_partial.events[-1][1] == 1 else 0
    recorder_partial = Recorder()
    partial_result = train.run(partial_args, clock=StopAfterFirstUpdate(), recorder=recorder_partial)
    assert partial_result["iter_num"] == 1 and partial_result["optimizer_updates"] == 1
    resumed_args = tiny_args(train, tmp_path, "partial", optimizer_name, "--dropout", ".15", "--dtype", dtype,
                             "--init-from", "resume")
    resumed_result = train.run(resumed_args)
    assert resumed_result["iter_num"] == 2 and resumed_result["data_batches"] == 4
    assert resumed_result["optimizer_updates"] == 2
    resumed = torch.load(tmp_path / "outputs/partial/ckpt.pt", weights_only=False)
    for key in ("model", "optimizer", "scaler", "rng", "data_batches", "iter_num", "optimizer_updates", "best_val_loss"):
        assert_tree_equal(full[key], resumed[key])
    # A completed run cannot be silently overwritten by a fresh initialization.
    with pytest.raises(ValueError, match="already contains"):
        train.run(full_args)
    assert not (tmp_path / "outputs/ckpt.pt").exists()
    assert not (tmp_path / "model.bin").exists()


@pytest.mark.parametrize("optimizer_name", ["HF", "AdamW"])
def test_run_amp_overflow_attempt_accounting_and_resume(lane, tmp_path, monkeypatch, capsys, optimizer_name):
    train, _, builder = lane
    tiny_data(tmp_path)
    original_transformer = train.Transformer
    initial_weights, initial_optimizer = [], []
    class OverflowTransformer(original_transformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            next(self.parameters()).register_hook(lambda grad: torch.full_like(grad, float("inf")))
            initial_weights.append(copy.deepcopy(self.state_dict()))
    original_build = builder.build_optimizer
    def record_build(*args, **kwargs):
        optimizer = original_build(*args, **kwargs)
        initial_optimizer.append(copy.deepcopy(optimizer.state_dict()))
        return optimizer
    monkeypatch.setattr(train, "Transformer", OverflowTransformer)
    monkeypatch.setattr(builder, "build_optimizer", record_build)
    def args(*extra):
        return tiny_args(train, tmp_path, "overflow", optimizer_name, "--dtype", "float16",
                         "--eval-interval", "0", *extra)
    recorder = Recorder()
    result = train.run(args(), recorder=recorder)
    assert result["iter_num"] == 2 and result["data_batches"] == 4
    assert result["optimizer_updates"] == 0 and result["optimizer_step_skips"] == 2
    assert result["data_batches"] * 2 * 4 == 2 * result["config"]["tokens_per_iteration"]
    checkpoint_path = tmp_path / "outputs/overflow/ckpt.pt"
    saved = torch.load(checkpoint_path, weights_only=False)
    assert saved["format_version"] == 2 and saved["optimizer_updates"] == 0
    assert saved["iter_num"] == 2 and saved["data_batches"] == 4
    assert saved["scaler"]["scale"] == 16384  # two real GradScaler backoffs from 65536
    assert_tree_equal(saved["model"], initial_weights[0])
    assert_tree_equal(saved["optimizer"], initial_optimizer[0])
    assert len(recorder.events) == 2
    for attempt, (metrics, step) in enumerate(recorder.events, 1):
        assert metrics["iter"] == step == attempt
        assert metrics["optimizer_updates"] == 0 and metrics["optimizer_step_skips"] == attempt
        assert metrics["optimizer_step_skipped"] is True
    console = capsys.readouterr().out
    assert "attempt 2: optimizer_updates=0, optimizer_step_skipped=True" in console
    assert "step 2:" not in console

    # Resume with finite gradients: data/attempts continue, and actual successful
    # optimizer calls are counted even though these wrappers return None.
    monkeypatch.setattr(train, "Transformer", original_transformer)
    for limit in (3, 4):
        resumed_recorder = Recorder()
        resumed = train.run(args("--init-from", "resume", "--max-iters", str(limit)),
                            recorder=resumed_recorder)
        assert resumed["iter_num"] == limit and resumed["data_batches"] == 2 * limit
        assert resumed["optimizer_updates"] == limit - 2 and resumed["optimizer_step_skips"] == 2
        continued = torch.load(checkpoint_path, weights_only=False)
        assert continued["optimizer_updates"] == limit - 2 and continued["iter_num"] == limit
        assert continued["scaler"]["scale"] >= saved["scaler"]["scale"]
        assert any(not torch.equal(saved["model"][k], value) for k, value in continued["model"].items())
        assert len(resumed_recorder.events) == 1
        metrics, step = resumed_recorder.events[0]
        assert step == limit and metrics["optimizer_updates"] == limit - 2
        assert metrics["optimizer_step_skipped"] is False and metrics["optimizer_step_skips"] == 2

    # Older checkpoints cannot supply a trustworthy successful-update history.
    for version in (1, 2):
        unsupported = copy.deepcopy(continued)
        unsupported["format_version"] = version
        unsupported.pop("optimizer_updates")
        legacy = tmp_path / f"missing-updates-v{version}.pt"
        torch.save(unsupported, legacy)
        with pytest.raises(ValueError, match="update accounting.*warmstart"):
            train.run(args("--init-from", "resume", "--checkpoint", str(legacy)))
    warm = train.run(args("--init-from", "warmstart", "--checkpoint", str(legacy),
                          "--run-name", "fresh-history", "--max-iters", "0"))
    assert warm["iter_num"] == warm["optimizer_updates"] == warm["optimizer_step_skips"] == 0


@pytest.mark.parametrize("optimizer_name", ["HF", "AdamW"])
def test_run_permanent_overflow_at_zero_scale_never_counts_updates(lane, tmp_path, monkeypatch, optimizer_name):
    train, _, builder = lane
    tiny_data(tmp_path)
    original_transformer, original_build = train.Transformer, builder.build_optimizer
    initial_weights, initial_optimizer, invoked_steps = [], [], []
    class OverflowTransformer(original_transformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            next(self.parameters()).register_hook(lambda grad: torch.full_like(grad, float("inf")))
            initial_weights.append(copy.deepcopy(self.state_dict()))
    def observe_build(*args, **kwargs):
        optimizer = original_build(*args, **kwargs)
        initial_optimizer.append(copy.deepcopy(optimizer.state_dict()))
        original_step = optimizer.step
        def observed_step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            invoked_steps.append(True)
            return result
        optimizer.step = observed_step
        return optimizer
    monkeypatch.setattr(train, "Transformer", OverflowTransformer)
    monkeypatch.setattr(builder, "build_optimizer", observe_build)
    recorder = Recorder()
    result = train.run(tiny_args(train, tmp_path, "zero-scale", optimizer_name, "--dtype", "float16",
                       "--max-iters", "170", "--eval-interval", "0"), recorder=recorder)
    saved = torch.load(tmp_path / "outputs/zero-scale/ckpt.pt", weights_only=False)
    assert saved["scaler"]["scale"] == 0
    assert invoked_steps == []
    assert_tree_equal(saved["model"], initial_weights[0])
    assert_tree_equal(saved["optimizer"], initial_optimizer[0])
    assert result["iter_num"] == saved["iter_num"] == 170
    assert result["data_batches"] == saved["data_batches"] == 340
    assert result["optimizer_updates"] == saved["optimizer_updates"] == 0
    assert result["optimizer_step_skips"] == 170
    if optimizer_name == "HF":
        assert saved["optimizer"]["state"]["counter"] == -1
    assert saved["format_version"] == 2
    assert len(recorder.events) == 170
    for attempt, (metrics, step) in enumerate(recorder.events, 1):
        assert metrics["iter"] == step == attempt
        assert metrics["optimizer_updates"] == 0 and metrics["optimizer_step_skips"] == attempt
        assert metrics["optimizer_step_skipped"] is True


def test_exact_resume_rejects_changed_configuration_and_workers(lane, tmp_path):
    train, _, _ = lane
    tiny_data(tmp_path)
    args = tiny_args(train, tmp_path, "config", "HF", "--max-iters", "0")
    train.run(args)
    with pytest.raises(ValueError, match="configuration differs"):
        train.run(tiny_args(train, tmp_path, "config", "HF", "--init-from", "resume", "--seed", "999"))
    shard = tmp_path / "data/tok32/shard1.bin"
    original = shard.read_bytes()
    shard.write_bytes(original[2:] + original[:2])
    with pytest.raises(ValueError, match="configuration differs"):
        train.run(tiny_args(train, tmp_path, "config", "HF", "--init-from", "resume"))
    shard.write_bytes(original)
    train.run(tiny_args(train, tmp_path, "workers", "HF", "--num-workers", "1", "--max-iters", "0"))
    state = torch.load(tmp_path / "outputs/workers/ckpt.pt", weights_only=False)
    assert not state["exact_resume"] and not state["config"]["exact_resume_supported"]
    with pytest.raises(ValueError, match="warmstart"):
        train.run(tiny_args(train, tmp_path, "workers", "HF", "--init-from", "resume"))


def test_zero_budget_final_checkpoint_and_legacy_warmstart(lane, tmp_path, monkeypatch):
    train, _, _ = lane
    monkeypatch.setenv("RANK", "-1")  # the non-DDP sentinel must still save rank-zero artifacts
    tiny_data(tmp_path)
    args = tiny_args(train, tmp_path, "empty", "AdamW", "--max-time", "00:00:00")
    assert train.run(args)["iter_num"] == 0
    checkpoint = tmp_path / "outputs/empty/ckpt.pt"
    state = torch.load(checkpoint, weights_only=False)
    legacy = tmp_path / "legacy.pt"
    torch.save({"model": state["model"], "model_args": state["model_args"]}, legacy)
    with pytest.raises(ValueError, match="warmstart"):
        train.run(tiny_args(train, tmp_path, "legacy-resume", "AdamW", "--init-from", "resume",
                            "--checkpoint", str(legacy)))
    result = train.run(tiny_args(train, tmp_path, "legacy-warm", "AdamW", "--init-from", "warmstart",
                                "--checkpoint", str(legacy), "--max-iters", "0"))
    assert result["iter_num"] == 0


def test_seed_and_replayed_data_order(lane, tmp_path):
    train, data, _ = lane
    tiny_data(tmp_path)
    train.seed_everything(3)
    first = (random.random(), np.random.random(), torch.rand(1), transformer(train).state_dict())
    train.seed_everything(3)
    second = (random.random(), np.random.random(), torch.rand(1), transformer(train).state_dict())
    assert_tree_equal(first, second)
    train.seed_everything(3, rank=1)
    assert random.random() != first[0]
    def stream(seed=3, rank=0, workers=0):
        return data.Task.iter_batches(2, "cpu", workers, split="train", max_seq_len=4,
                                     vocab_size=32, vocab_source="custom", seed=seed,
                                     data_dir=tmp_path / "data", rank=rank)
    original = train.ReplayStream(stream())
    for _ in range(5):
        next(original)
    replayed = train.ReplayStream(stream(), consumed=5)
    assert_tree_equal(next(original), next(replayed))
    default = next(stream())
    assert not torch.equal(default[0], next(stream(seed=4))[0])
    assert not torch.equal(default[0], next(stream(rank=1))[0])
    for workers in (0, 2):
        assert_tree_equal(next(stream(workers=workers)), next(stream(workers=workers)))


def test_budget_interrupt_discards_partial_gradients(lane):
    train, _, _ = lane
    model = transformer(train)
    class NeverStep:
        def zero_grad(self, set_to_none=True):
            model.zero_grad(set_to_none=set_to_none)
        def step(self, *args, **kwargs):
            pytest.fail("Interrupted accumulation must not update")
    x = torch.zeros((1, 4), dtype=torch.long)
    stream = train.ReplayStream(iter([(x, x), (x, x)]))
    checks = iter([False, False, True])
    result = train.accumulated_step(model, model, NeverStep(), stream, 2,
               torch.amp.GradScaler("cpu", enabled=False), 1, expired=lambda: next(checks))
    assert result is None and stream.consumed == 1
    assert all(p.grad is None for p in model.parameters())
    ticks = iter([10, 10.5, 11])
    budget = train.TimeBudget("00:00:01", clock=lambda: next(ticks))
    assert not budget.expired() and budget.expired()


def test_attention_modes_and_cli_controls(lane):
    train, _, _ = lane
    train.seed_everything(2)
    manual = transformer(train)
    params = copy.deepcopy(manual.params)
    params.attention = "sdpa"
    sdpa = train.Transformer(params)
    sdpa.load_state_dict({k: v for k, v in manual.state_dict().items() if not k.endswith(".mask")})
    x = torch.randint(32, (2, 4))
    torch.testing.assert_close(manual(x, x), sdpa(x, x), atol=1e-6, rtol=1e-5)
    assert not manual.layers[0].attention.flash and sdpa.layers[0].attention.flash
    args = train.parse_args(["--max-iters", "7", "--no-compile", "--attention", "sdpa"])
    assert args.total_steps == 7 and not args.compile and args.attention == "sdpa"
    with pytest.raises(ValueError, match="single directory"):
        train.parse_args(["--run-name", "../escape"])
    with pytest.raises(ValueError, match="single process"):
        train.run(train.parse_args(["--init-from", "resume", "--num-workers", "1"]))


@pytest.fixture
def json_shards(tmp_path):
    directory = tmp_path / "TinyStories_all_data"
    directory.mkdir()
    for i in range(2):
        (directory / f"shard{i}.json").write_text(json.dumps([
            {"story": "Once upon a time, a little cat found a blue ball."},
            {"story": "The happy cat played all day and went home."},
        ]))
    return directory


def test_real_json_tokenization_manifest(lane, tmp_path, json_shards, capfd):
    _, data, _ = lane
    tokenizer = SOURCE / "tokenizer.model"  # local immutable SentencePiece fixture
    manifest = data.pretokenize(0, data_dir=tmp_path, tokenizer_model=str(tokenizer), max_workers=2)
    assert "Done." in capfd.readouterr().out
    assert len(manifest["shards"]) == 2
    assert manifest["tokenizer_sha256"] == data.file_sha256(tokenizer)
    expected = data.Tokenizer(str(tokenizer)).encode(
        "Once upon a time, a little cat found a blue ball.", bos=True, eos=False)
    tokens = np.fromfile(json_shards / "shard0.bin", dtype=np.uint16)
    np.testing.assert_array_equal(tokens[:len(expected)], expected)
    assert json.loads((json_shards / "manifest.json").read_text()) == manifest
    # A success manifest shares the Llama2 shard directory, but is not input.
    assert data.pretokenize(0, data_dir=tmp_path, tokenizer_model=str(tokenizer), max_workers=1) == manifest
    assert not list(json_shards.glob("*.tmp"))
    train = lane[0]
    args = train.parse_args(["--data-dir", str(tmp_path), "--tokenizer-model", str(tokenizer)])
    identity = train.dataset_identity(args)
    assert identity["tokenizer_sha256"] == manifest["tokenizer_sha256"]
    assert identity["tokenizer_status"] == "local"
    shard = json_shards / "shard1.bin"
    with shard.open("ab") as target:
        target.write(b"\\x01\\x00")
    with pytest.raises(ValueError, match="do not match preprocessing manifest"):
        train.dataset_identity(args)


def test_two_shard_worker_failure_has_no_success_manifest(lane, tmp_path, json_shards, capfd):
    _, data, _ = lane
    (json_shards / "shard1.json").write_text('{"invalid json')
    (json_shards / "manifest.json").write_text('{"old": "success"}')
    with pytest.raises(json.JSONDecodeError):
        data.pretokenize(0, data_dir=tmp_path, tokenizer_model=str(SOURCE / "tokenizer.model"), max_workers=2)
    assert "Done." not in capfd.readouterr().out
    assert not (json_shards / "manifest.json").exists()


def test_preprocessing_rejects_unexpected_shard(lane, tmp_path, json_shards):
    _, data, _ = lane
    np.array([1, 2], dtype=np.uint16).tofile(json_shards / "stale.bin")
    with pytest.raises(ValueError, match="unexpected token shard"):
        data.pretokenize(0, data_dir=tmp_path, tokenizer_model=str(SOURCE / "tokenizer.model"), max_workers=1)
    assert not (json_shards / "manifest.json").exists()
