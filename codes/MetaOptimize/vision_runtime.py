"""Shared vision runtime: single-process, completed-epoch checkpoints are resumable.

Time budgets apply to each invocation, include setup, and are checked between
batches (an in-flight batch / checkpoint write is not preempted). Distributed
checkpoints and interrupted epochs are weight-only warmstarts, not exact replay.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed, deterministic=False):
    # Required by deterministic CUDA matrix multiplication, before CUDA use.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    """DataLoader gives each worker a reproducible torch seed each epoch."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def time_seconds(value):
    try:
        hours, minutes, seconds = map(int, value.split(":"))
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, AttributeError):
        raise ValueError("max-time must be nonnegative hh:mm:ss") from None


class Budget:
    def __init__(self, seconds, clock=None, distributed=False, device="cpu"):
        self.clock = clock or time.monotonic
        self.start = self.clock()
        self.seconds = seconds
        self.distributed = distributed
        self.device = device

    @property
    def elapsed(self):
        return self.clock() - self.start

    def expired(self):
        stop = self.elapsed >= self.seconds
        if self.distributed:
            flag = torch.tensor(int(stop), device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())
        return stop


def prepare_run(save_directory, run_name):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a single nonempty directory name")
    path = Path(save_directory) / run_name
    # Even a resume writes to a fresh run-name: never overwrite a previous run.
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Run directory already exists: {path}; choose a new --run-name") from None
    return path


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def dataset_info(dataset):
    info = {"type": type(dataset).__name__, "samples": len(dataset)}
    for name in ("root", "split", "train", "classes", "class_to_idx", "random_offset"):
        if hasattr(dataset, name):
            info[name] = getattr(dataset, name)
    if "root" in info:
        info["root"] = str(Path(info["root"]).resolve())
    if hasattr(dataset, "samples"):
        # Fingerprint the path/label index, not image contents (no startup scan).
        digest = hashlib.sha256()
        for path, label in dataset.samples:
            digest.update(f"{path}\0{label}\n".encode())
        info["path_label_sha256"] = digest.hexdigest()
    if hasattr(dataset, "targets"):
        info["labels_sha256"] = hashlib.sha256(np.asarray(dataset.targets, dtype=np.int64).tobytes()).hexdigest()
    info["transform"] = repr(getattr(dataset, "transform", None))
    return info


def make_loader(dataset, batch_size, seed, workers=0, shuffle=False, sampler=None):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
        sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=generator, persistent_workers=False)


def rng_state(loaders):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "loaders": [loader.generator.get_state() for loader in loaders]}


def restore_rng(state, loaders):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA for exact resume")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state.get("mps") is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS checkpoint requires MPS for exact resume")
        torch.mps.set_rng_state(state["mps"].cpu())
    if len(state["loaders"]) != len(loaders):
        raise ValueError("checkpoint loader count mismatch")
    for loader, saved in zip(loaders, state["loaders"]):
        loader.generator.set_state(saved.cpu())


def unwrap(model):
    return model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model


def atomic_save(state, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_weights(path, model, device):
    # Only load trusted local checkpoints: full RNG state uses Python objects.
    state = torch.load(path, map_location=device, weights_only=False)
    weights = state.get("model", state.get("state_dict", state))
    if weights and all(key.startswith("module.") for key in weights):
        weights = {key[7:]: value for key, value in weights.items()}
    unwrap(model).load_state_dict(weights)


def resume_checkpoint(path, model, optimizer, loaders, signature, device, distributed=False):
    state = torch.load(path, map_location=device, weights_only=False)
    if distributed or state.get("format_version") != 1 or not state.get("restartable", False):
        raise ValueError("--resume requires a full single-process completed-epoch checkpoint; use --warmstart for weights only")
    if state["signature"] != signature:
        raise ValueError("resume configuration/dataset mismatch; use --warmstart for a different experiment")
    unwrap(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], loaders)
    return state


def epoch_pass(loader, model, criterion, device, optimizer=None, budget=None, print_freq=0):
    """Mean criterion losses are weighted by sample count, including short batches."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = correct5 = samples = updates = 0
    complete = True
    with torch.enable_grad() if training else torch.no_grad():
        iterator = iter(loader)
        for batch_index in range(len(loader)):
            if budget is not None and budget.expired():
                complete = False
                break
            inputs, targets = next(iterator)
            # Loading/augmentation can consume the remaining budget. Do not
            # launch a new update after that work has already exhausted it.
            if budget is not None and budget.expired():
                complete = False
                break
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step(model)
                updates += 1
            count = targets.size(0)
            loss_sum += float(loss.detach()) * count
            correct += int((outputs.detach().argmax(1) == targets).sum())
            top5 = outputs.detach().topk(min(5, outputs.size(1)), dim=1).indices
            correct5 += int(top5.eq(targets[:, None]).any(dim=1).sum())
            samples += count
            if print_freq > 0 and batch_index % print_freq == 0:
                print(f"Batch {batch_index}: loss={loss_sum / samples:.6f}, samples={samples}")
    totals = torch.tensor([loss_sum, correct, correct5, samples], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    loss_sum, correct, correct5, samples = totals.tolist()
    return {"loss": loss_sum / samples if samples else None,
            "accuracy": 100 * correct / samples if samples else None,
            "accuracy5": 100 * correct5 / samples if samples else None,
            "samples": int(samples), "updates": updates, "complete": complete}


def run_training(args, model, optimizer, train_loader, val_loader, writer, run_dir,
                 budget, architecture, epochs, device, start_epoch=0):
    distributed = getattr(args, "distributed", False)
    primary = not distributed or dist.get_rank() == 0
    loaders = [train_loader, val_loader]
    configuration = optimizer.config
    # Exact resume is intentionally restricted to the same runtime/data settings.
    signature = {
        "optimizer": configuration, "architecture": architecture,
        "datasets": [dataset_info(loader.dataset) for loader in loaders],
        "batch_size": args.batch_size, "workers": getattr(args, "workers", 0),
        "seed": args.seed, "deterministic": args.deterministic,
        "device": str(device), "precision": "float32", "torch_version": str(torch.__version__),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    manifest = {"args": {key: str(value) if isinstance(value, (torch.device, Path)) else value
                         for key, value in vars(args).items()},
                **signature, "scheduler": configuration.get("scheduler"),
                "compile": False, "attention": "architecture default",
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else str(device),
                "distributed": distributed,
                "world_size": dist.get_world_size() if distributed else 1,
                "batch_size_per_process": args.batch_size,
                "counter_semantics": "samples count training examples across ranks (including sampler padding); updates count synchronized optimizer steps",
                "resume_support": "single-process completed epochs only; partial/distributed states are weight-only",
                "budget_semantics": "per invocation including setup; checked before train/validation batches; in-flight work and saving may overrun",
                "determinism_tradeoff": "deterministic algorithms may be slower or raise for unsupported operations" if args.deterministic else "performance mode; seeds alone do not guarantee CUDA determinism"}
    if primary:
        write_json(run_dir / "manifest.json", manifest)
    best_accuracy = -1.0
    samples = updates = 0
    prior_elapsed = 0.0
    if args.resume:
        state = resume_checkpoint(args.resume, model, optimizer, loaders, signature, device, distributed)
        start_epoch = state["next_epoch"]
        best_accuracy = state["best_accuracy"]
        samples, updates = state["samples"], state["updates"]
        prior_elapsed = state["total_elapsed_seconds"]
    elif args.warmstart:
        load_weights(args.warmstart, model, device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    boundary = True
    reason = "max_epochs"
    next_epoch = start_epoch

    def checkpoint(is_best=False):
        if not primary:
            return
        state = {"format_version": 1, "model": unwrap(model).state_dict(),
                 "optimizer": optimizer.state_dict(), "rng": rng_state(loaders),
                 "signature": signature, "next_epoch": next_epoch,
                 "best_accuracy": best_accuracy, "samples": samples, "updates": updates,
                 "elapsed_seconds": budget.elapsed,
                 "total_elapsed_seconds": prior_elapsed + budget.elapsed,
                 "stop_reason": reason, "epoch_boundary": boundary,
                 "restartable": boundary and not distributed,
                 "restart_note": "single-process completed-epoch only; otherwise use --warmstart"}
        atomic_save(state, run_dir / "checkpoint.pt")
        if is_best:
            atomic_save(state, run_dir / "best.pt")
        return state

    for epoch in range(start_epoch, epochs):
        if budget.expired():
            reason = "budget"
            break
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        result = epoch_pass(train_loader, model, criterion, device, optimizer, budget,
                            print_freq=getattr(args, "print_freq", 0))
        samples += result["samples"]
        updates += result["updates"]
        boundary = False
        if primary and result["samples"]:
            loss_tag = "Performance/train_losses" if hasattr(args, "arch") else "Performance/train_loss"
            accuracy_tag = "Performance/train_acc1" if hasattr(args, "arch") else "Performance/train_accuracy"
            writer.add_scalar(loss_tag, result["loss"], epoch)
            writer.add_scalar(accuracy_tag, result["accuracy"], epoch)
            if hasattr(args, "arch"):
                writer.add_scalar("Performance/train_acc5", result["accuracy5"], epoch)
            for group, lr in enumerate(optimizer.get_learning_rates()):
                writer.add_scalar(f"Learning_rate/group_{group}", lr, updates)
        if not result["complete"] or budget.expired():
            reason = "budget"
            break
        validation = epoch_pass(val_loader, model, criterion, device, budget=budget)
        if not validation["complete"]:
            reason = "budget"
            break
        boundary = True
        next_epoch = epoch + 1
        accuracy = validation["accuracy"]
        is_best = accuracy is not None and accuracy > best_accuracy
        if accuracy is not None:
            best_accuracy = max(best_accuracy, accuracy)
            if primary:
                tag = "Performance/val_acc1" if hasattr(args, "arch") else "Performance/test_accuracy"
                writer.add_scalar(tag, accuracy, epoch)
        if primary and getattr(args, "verbos", 1):
            print(f"Epoch {epoch}: samples={result['samples']}, train_loss={result['loss']}, validation_accuracy={accuracy}")
        reason = "epoch_complete"
        checkpoint(is_best)
    else:
        reason = "max_epochs"
    state = checkpoint()
    if primary:
        summary = {key: state[key] for key in ("samples", "updates", "elapsed_seconds", "total_elapsed_seconds", "stop_reason", "epoch_boundary", "restartable", "next_epoch")}
        write_json(run_dir / "summary.json", summary)
    return state
