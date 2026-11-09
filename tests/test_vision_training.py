"""Offline vision regression tests: real Torch updates, no downloads or GPUs."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1] / "codes" / "MetaOptimize"
sys.path.insert(0, str(ROOT))
import vision_runtime as runtime


def script(name):
    if name == "imagenet":
        pytest.importorskip("torchvision", reason="ImageNet tests require the experiments extra (torchvision)",
                            exc_type=ModuleNotFoundError)
    spec = importlib.util.spec_from_file_location(f"vision_{name}", ROOT / name / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Recorder:
    def __init__(self, *args):
        self.values = []
        self.closed = False

    def add_scalar(self, *args):
        self.values.append(args)

    def close(self):
        self.closed = True


class TorchProtocol:
    """Real Torch optimizer/scheduler using the shared experiment protocol."""
    def __init__(self, model, lr=0.04):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.97)
        self.config = {"optimizer": "SGDm", "lr": lr,
                       "scheduler": {"type": "StepLR", "gamma": 0.97}}

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, model, loss=None):
        if loss is not None:
            self.zero_grad()
            loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def get_learning_rates(self):
        return [group["lr"] for group in self.param_groups]

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


def tiny_data():
    return TensorDataset(torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10,
                         torch.tensor([0, 1, 0, 1, 1]))


def tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Dropout(0.2), torch.nn.Linear(5, 2))


def run(tmp_path, name, epochs=1, resume="", budget=None, imagenet=False,
        optimizer_factory=TorchProtocol, workers=0, dataset_factory=tiny_data):
    runtime.seed_everything(7, True)
    model = tiny_model()
    optimizer = optimizer_factory(model)
    args = argparse.Namespace(batch_size=2, seed=7, deterministic=True, workers=workers,
                              resume=str(resume), warmstart="", verbos=0)
    if imagenet:
        args.arch = "tiny"
    loaders = [runtime.make_loader(dataset_factory(), 2, 7, workers=workers, shuffle=True),
               runtime.make_loader(dataset_factory(), 2, 8, workers=workers)]
    writer = Recorder()
    directory = runtime.prepare_run(tmp_path, name)
    state = runtime.run_training(args, model, optimizer, *loaders, writer, directory,
                                 budget or runtime.Budget(1000), "tiny", epochs, "cpu")
    return state, writer, directory


@pytest.mark.parametrize("imagenet", [False, True])
def test_sample_weighted_epoch_loss_and_recorder(tmp_path, imagenet):
    dataset = TensorDataset(torch.tensor([[4., 0.], [4., 0.], [0., 4.]]), torch.zeros(3, dtype=torch.long))
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    optimizer = TorchProtocol(model, lr=0.)
    args = argparse.Namespace(seed=1, deterministic=True, workers=0, batch_size=2,
                              resume="", warmstart="", verbos=0)
    if imagenet:
        args.arch = "tiny"
    loader = runtime.make_loader(dataset, 2, 1)
    recorder = Recorder()
    directory = runtime.prepare_run(tmp_path, "loss")
    runtime.run_training(args, model, optimizer, loader, loader, recorder, directory,
                         runtime.Budget(100), "tiny", 1, "cpu")
    expected = torch.nn.functional.cross_entropy(dataset.tensors[0], dataset.tensors[1]).item()
    tag = "Performance/train_losses" if imagenet else "Performance/train_loss"
    assert next(value for key, value, step in recorder.values if key == tag) == pytest.approx(expected)
    assert not np.isclose(expected, sum(torch.nn.functional.cross_entropy(x, y).item()
                                       for x, y in loader) / len(loader))


def test_seed_model_data_and_workers():
    def snapshot(seed):
        runtime.seed_everything(seed, True)
        model = tiny_model()
        values = [model[0].weight.detach().clone(), random.random(), np.random.rand()]
        loader = runtime.make_loader(tiny_data(), 2, seed, workers=2, shuffle=True)
        values.append(torch.cat([x for x, y in loader]))
        return values
    a, b, c = snapshot(4), snapshot(4), snapshot(5)
    assert torch.equal(a[0], b[0]) and a[1:3] == b[1:3] and torch.equal(a[3], b[3])
    assert not torch.equal(a[0], c[0]) and not torch.equal(a[3], c[3])


def test_worker_python_numpy_and_torch_seeds():
    class RandomDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return torch.tensor([random.random(), np.random.rand(), torch.rand(()).item()]), index

    def draws(seed):
        return torch.cat([x for x, y in runtime.make_loader(RandomDataset(), 2, seed, workers=2)])
    assert torch.equal(draws(1), draws(1))
    assert not torch.equal(draws(1), draws(2))


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


def test_budget_stops_inside_epoch_and_saves_partial(tmp_path):
    clock = Clock()
    budget = runtime.Budget(1, clock=clock)
    original = TorchProtocol.step
    # Simulate a batch taking longer than the budget without sleeping.
    def slow_step(self, model, loss=None):
        original(self, model, loss)
        clock.now += 2
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TorchProtocol, "step", slow_step)
        state, _, directory = run(tmp_path, "budget", budget=budget)
    assert state["updates"] == 1 and state["samples"] == 2
    assert state["stop_reason"] == "budget" and state["elapsed_seconds"] == 2
    assert not state["epoch_boundary"] and not state["restartable"]
    assert (directory / "checkpoint.pt").exists()
    assert json.loads((directory / "summary.json").read_text())["updates"] == 1
    with pytest.raises(ValueError, match="--warmstart"):
        run(tmp_path, "invalid", resume=directory / "checkpoint.pt")


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_completed_epoch_resume_matches_uninterrupted_model_optimizer_scheduler_rng(tmp_path):
    full, _, _ = run(tmp_path, "full", epochs=2)
    first, _, first_dir = run(tmp_path, "first", epochs=1)
    resumed, _, _ = run(tmp_path, "resumed", epochs=2, resume=first_dir / "checkpoint.pt")
    assert first["restartable"] and resumed["restartable"]
    assert resumed["next_epoch"] == 2 and resumed["updates"] == full["updates"] == 6
    assert resumed["samples"] == full["samples"] == 10
    for key in ("model", "optimizer"):
        assert_nested_equal(full[key], resumed[key])
    assert_nested_equal(full["rng"], resumed["rng"])


def test_run_paths_manifest_and_legacy_warmstart(tmp_path):
    state, _, directory = run(tmp_path, "one")
    _, _, other = run(tmp_path, "two", epochs=0)
    assert directory != other
    assert (directory / "best.pt").exists()
    with pytest.raises(FileExistsError):
        runtime.prepare_run(tmp_path, "one")
    with pytest.raises(ValueError):
        runtime.prepare_run(tmp_path, "../escape")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["optimizer"]["optimizer"] == "SGDm"
    assert manifest["scheduler"]["gamma"] == 0.97
    assert manifest["precision"] == "float32" and manifest["device"] == "cpu"
    assert manifest["datasets"][0]["samples"] == 5 and manifest["compile"] is False
    assert state["stop_reason"] == "max_epochs"
    legacy = tmp_path / "legacy.pt"
    torch.save({"state_dict": state["model"]}, legacy)
    with pytest.raises(ValueError, match="--warmstart"):
        run(tmp_path, "legacy-resume", resume=legacy)
    model = tiny_model()
    runtime.load_weights(legacy, model, "cpu")
    assert_nested_equal(model.state_dict(), state["model"])


@pytest.mark.parametrize("lane", ["cifar10", "imagenet"])
def test_cli_and_tiny_cpu_main_without_downloads(tmp_path, monkeypatch, lane):
    pytest.importorskip("tensorboard", reason="Training entrypoint tests require the experiments extra (tensorboard)",
                        exc_type=ModuleNotFoundError)
    module = script(lane)
    recorder = Recorder()
    import torch.utils.tensorboard
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", lambda *a: recorder)
    def builder(model, args, writer):
        assert args.total_steps == 3 and args.warmup_steps == 0
        assert args.base_lr == .01 and args.base_momentum == .9
        return TorchProtocol(model)
    monkeypatch.setitem(sys.modules, "Optimizers.build_optimizer", types.SimpleNamespace(build_optimizer=builder))
    argv = ["--save-directory", str(tmp_path), "--run-name", lane, "--batch-size", "2",
            "--workers", "0", "--deterministic", "--max-time", "00:01:00"]
    if lane == "cifar10":
        monkeypatch.setitem(sys.modules, "build_network", types.SimpleNamespace(build_network=lambda *a: tiny_model()))
        loader = runtime.make_loader(tiny_data(), 2, 1)
        monkeypatch.setitem(sys.modules, "load_data", types.SimpleNamespace(load_data=lambda *a: (loader, loader)))
        args = module.parse_args([])
        argv += ["--num-epochs", "1"]
    else:
        monkeypatch.setitem(module.models.__dict__, "resnet18", lambda **kwargs: tiny_model())
        monkeypatch.setattr(module, "load_datasets", lambda args: (tiny_data(), tiny_data()))
        args = module.parser.parse_args([])
        argv += ["--epochs", "1"]
    assert args.optimizer == "HF" and args.alg_meta == "Lion"
    module.main(argv)
    assert recorder.closed
    saved = torch.load(tmp_path / lane / "checkpoint.pt", weights_only=False)
    assert saved["updates"] == 3 and saved["samples"] == 5
    assert saved["restartable"]


def test_cifar_writer_closed_on_setup_error(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard", reason="Writer lifecycle tests require the experiments extra (tensorboard)",
                        exc_type=ModuleNotFoundError)
    module = script("cifar10")
    recorder = Recorder()
    import torch.utils.tensorboard
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", lambda *a: recorder)
    def fail(*args):
        raise RuntimeError("setup failed")
    monkeypatch.setitem(sys.modules, "build_network", types.SimpleNamespace(build_network=fail))
    monkeypatch.setitem(sys.modules, "load_data", types.SimpleNamespace(load_data=None))
    monkeypatch.setitem(sys.modules, "Optimizers.build_optimizer", types.SimpleNamespace(build_optimizer=None))
    with pytest.raises(RuntimeError, match="setup failed"):
        module.main(["--save-directory", str(tmp_path), "--run-name", "failure"])
    assert recorder.closed


@pytest.mark.parametrize("value, seconds", [("24:00:00", 86400), ("00:00:00", 0)])
def test_existing_time_format(value, seconds):
    assert runtime.time_seconds(value) == seconds


@pytest.mark.parametrize("value", ["-1:00:00", "00:60:00", "5", "00:00:-1"])
def test_invalid_time_format(value):
    with pytest.raises(ValueError, match="hh:mm:ss"):
        runtime.time_seconds(value)


def test_budget_rechecked_after_loading_before_update():
    clock = Clock()
    class SlowDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            clock.now += 1
            return torch.ones(2), 0
    model = tiny_model()
    result = runtime.epoch_pass(runtime.make_loader(SlowDataset(), 2, 1), model,
                                torch.nn.CrossEntropyLoss(), "cpu", TorchProtocol(model),
                                runtime.Budget(1, clock=clock))
    assert not result["complete"] and result["updates"] == result["samples"] == 0


def test_zero_budget_still_saves_restartable_final_state(tmp_path):
    state, _, directory = run(tmp_path, "zero", budget=runtime.Budget(0))
    assert state["stop_reason"] == "budget" and state["restartable"]
    assert state["samples"] == state["updates"] == state["next_epoch"] == 0
    assert (directory / "checkpoint.pt").exists()


def test_resume_rejects_changed_configuration(tmp_path):
    _, _, directory = run(tmp_path, "source")
    path = directory / "checkpoint.pt"
    state = torch.load(path, weights_only=False)
    state["signature"]["seed"] += 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="configuration/dataset mismatch"):
        run(tmp_path, "changed", resume=path)


@pytest.mark.parametrize("lane, name", [("cifar10", name) for name in ("AdamW", "HF")] +
                         [("imagenet", name) for name in ("SGD", "SGDm", "Adam", "AdamW", "Lion", "HF")])
def test_real_builder_paths_update_and_full_epoch_resume(tmp_path, monkeypatch, lane, name):
    source = ROOT / lane / "Optimizers"
    for key in list(sys.modules):
        if key == "Optimizers" or key.startswith("Optimizers."):
            monkeypatch.delitem(sys.modules, key)
    package = types.ModuleType("Optimizers")
    package.__path__ = [str(source)]
    monkeypatch.setitem(sys.modules, "Optimizers", package)
    builder = importlib.import_module("Optimizers.build_optimizer").build_optimizer
    module = script(lane)
    args = module.parse_args([]) if lane == "cifar10" else module.parser.parse_args([])
    args.optimizer = name
    args.total_steps, args.warmup_steps, args.alpha0 = 12, 1, .02
    args.base_lr, args.base_momentum, args.weight_decay_base = .05, .83, .12
    for prefix in ("normalizer_param", "momentum_param", "weight_decay", "Lion_beta2"):
        for level in ("base", "meta"):
            key = f"{prefix}_{level}"
            if getattr(args, key) == -1:
                setattr(args, key, None)
    factory = lambda model: builder(model, args, Recorder())
    full, _, directory = run(tmp_path, "full", epochs=2, optimizer_factory=factory)
    _, _, first_dir = run(tmp_path, "first", epochs=1, optimizer_factory=factory)
    resumed, _, _ = run(tmp_path, "resumed", epochs=2, resume=first_dir / "checkpoint.pt",
                        optimizer_factory=factory)
    runtime.seed_everything(7, True)
    initial = tiny_model().state_dict()
    assert any(not torch.equal(initial[k], full["model"][k]) for k in initial)
    for key in ("model", "optimizer"):
        assert_nested_equal(full[key], resumed[key])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["optimizer"] == full["signature"]["optimizer"]
    if name != "HF":
        assert manifest["scheduler"] is not None
        expected_lr = args.base_lr if name in ("SGD", "SGDm", "Adam") else args.alpha0
        assert manifest["optimizer"]["lr"] == expected_lr
        assert manifest["optimizer"]["weight_decay"] == args.weight_decay_base
        if name == "SGDm":
            assert manifest["optimizer"]["momentum"] == args.base_momentum
    assert full["updates"] == resumed["updates"] == 6


@pytest.mark.parametrize("lane", ["cifar10", "imagenet"])
def test_real_resnet_cpu_synthetic_image_update(lane):
    runtime.seed_everything(3, True)
    if lane == "cifar10":
        spec = importlib.util.spec_from_file_location("cifar_build_network", ROOT / lane / "build_network.py")
        network = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(network)
        model = network.build_network("ResNet18", "cpu")
    else:
        pytest.importorskip("torchvision", reason="ImageNet ResNet tests require the experiments extra (torchvision)",
                            exc_type=ModuleNotFoundError)
        import torchvision.models
        model = torchvision.models.resnet18(weights=None)
    loader = runtime.make_loader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), 2, 3)
    before = next(model.parameters()).detach().clone()
    result = runtime.epoch_pass(loader, model, torch.nn.CrossEntropyLoss(), "cpu", TorchProtocol(model))
    assert result["updates"] == 1 and result["samples"] == 2 and result["complete"]
    assert not torch.equal(before, next(model.parameters()))


def test_imagenet_two_rank_cpu_ddp_training_entrypoint(tmp_path):
    pytest.importorskip("torchvision", reason="ImageNet DDP tests require the experiments extra (torchvision)",
                        exc_type=ModuleNotFoundError)
    pytest.importorskip("tensorboard", reason="ImageNet DDP tests require the experiments extra (tensorboard)",
                        exc_type=ModuleNotFoundError)
    entry = tmp_path / "distributed_vision.py"
    entry.write_text(f'''
import sys
sys.path[:0] = [{str(ROOT / "imagenet")!r}, {str(ROOT)!r}]
import torch
from torch.utils.data import TensorDataset
import train
train.models.__dict__["resnet18"] = lambda **kw: torch.nn.Linear(2, 2)
def data(args):
    dataset = TensorDataset(torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10,
                            torch.tensor([0, 1, 0, 1, 1]))
    return dataset, dataset
train.load_datasets = data
train.main()
''')
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", str(entry), "--dist-url", "env://", "--dist-backend", "gloo",
               "--save-directory", str(tmp_path), "--run-name", "ddp", "--epochs", "1",
               "--batch-size", "2", "--workers", "0", "--optimizer", "HF", "--deterministic"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    state = torch.load(tmp_path / "ddp" / "checkpoint.pt", weights_only=False)
    manifest = json.loads((tmp_path / "ddp" / "manifest.json").read_text())
    assert state["samples"] == 6 and state["updates"] == 2  # updates per rank
    assert state["epoch_boundary"] and not state["restartable"]
    assert manifest["distributed"] and "padding" in manifest["args"]["validation_sampler"]


@pytest.mark.parametrize("argv, message", [(["--start-epoch", "1"], "not a resume"),
    (["--evaluate", "--resume", "checkpoint.pt"], "--warmstart"),
    (["--world-size", "2", "--resume", "checkpoint.pt"], "distributed --resume")])
def test_imagenet_rejects_unsupported_resume_modes(argv, message):
    with pytest.raises(ValueError, match=message):
        script("imagenet").main(argv)


def test_imagenet_evaluation_records_budget_and_samples(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard", reason="Evaluation entrypoint tests require the experiments extra (tensorboard)",
                        exc_type=ModuleNotFoundError)
    module = script("imagenet")
    monkeypatch.setitem(module.models.__dict__, "resnet18", lambda **kw: tiny_model())
    monkeypatch.setattr(module, "load_datasets", lambda args: (tiny_data(), tiny_data()))
    module.main(["--evaluate", "--max-time", "00:00:00", "--save-directory", str(tmp_path),
                 "--run-name", "evaluate", "--workers", "0"])
    result = json.loads((tmp_path / "evaluate" / "evaluation.json").read_text())
    assert result["stop_reason"] == "budget" and result["samples"] == result["updates"] == 0
    assert result["elapsed_seconds"] >= 0
    manifest = json.loads((tmp_path / "evaluate" / "manifest.json").read_text())
    assert manifest["optimizer"] is None and manifest["scheduler"] is None


class AugmentedDataset(torch.utils.data.Dataset):
    """Exercise all three augmentation RNGs in main process or loader workers."""
    def __len__(self):
        return 5

    def __getitem__(self, index):
        noise = random.random() + np.random.rand() + torch.rand(()).item()
        return torch.tensor([index / 5, noise]), index % 2


@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_resume_replays_stochastic_augmentation_workers(tmp_path, workers):
    options = dict(workers=workers, dataset_factory=AugmentedDataset)
    full, _, _ = run(tmp_path, "full", epochs=2, **options)
    _, _, directory = run(tmp_path, "first", epochs=1, **options)
    resumed, _, _ = run(tmp_path, "resumed", epochs=2, resume=directory / "checkpoint.pt", **options)
    for key in ("model", "optimizer", "rng"):
        assert_nested_equal(full[key], resumed[key])
