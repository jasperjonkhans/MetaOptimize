"""TinyStories training with accumulated gradients and replay-based exact resume.

Exact resume is supported with one process and num_workers=0 only. Rebuilding the
stream replays every consumed microbatch (O(consumed batches) I/O); it does not
cache the dataset. DDP/worker checkpoints are explicitly warmstart-only.
max_time is a soft monotonic budget, checked between batches/updates, not a
preemptive timeout for an in-flight forward, collective, export or filesystem I/O.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Transformer, ModelArgs
from tinystories import Task, file_sha256, token_shards


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    options = {
        "seed": (int, 1), "run-name": (str, "test"),
        "max-time": (str, "24:00:00"), "save-directory": (str, "outputs"),
        "optimizer": (str, "HF"), "stepsize-groups": (str, "scalar"),
        "meta-stepsize": (float, 1e-3), "alpha0": (float, 1e-5), "gamma": (float, 1),
        "batch-size": (int, 64), "gradient-accumulation-steps": (int, 8),
        "max-seq-len": (int, 256), "vocab-size": (int, 32000),
        "dim": (int, 288), "n-layers": (int, 6), "n-heads": (int, 6),
        "n-kv-heads": (int, 6), "multiple-of": (int, 32), "dropout": (float, 0),
        "grad-clip": (float, 1),
        "warmup-steps": (int, 1000), "total-steps": (int, None),
        "device": (str, "cuda"), "data-dir": (str, "data"),
        "tokenizer-model": (str, None), "num-workers": (int, 0),
        "checkpoint": (str, None), "eval-interval": (int, 2000),
        "eval-iters": (int, 100), "log-interval": (int, 1),
        "wandb-project": (str, "llamac"),
    }
    parser.add_argument("--max-iters", type=int, default=100000,
                        help="Maximum attempted accumulated iterations, including AMP-skipped updates")
    for name, (kind, default) in options.items():
        parser.add_argument("--" + name, type=kind, default=default)
    for suffix, defaults in [("base", ("AdamW", .999, .9, .1, -1)),
                             ("meta", ("Lion", -1, .99, 0, .9))]:
        for name, value in zip(("alg", "normalizer-param", "momentum-param", "weight-decay", "Lion-beta2"), defaults):
            parser.add_argument(f"--{name}-{suffix}", type=str if name == "alg" else float, default=value)
    parser.add_argument("--vocab-source", choices=["llama2", "custom"], default="llama2")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--init-from", choices=["scratch", "resume", "warmstart"], default="scratch")
    parser.add_argument("--attention", choices=["manual", "sdpa"], default="manual")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--wandb-log", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    return parser


def parse_args(argv=None):
    args = make_parser().parse_args(argv)
    for name in vars(args):
        if name.endswith(("_base", "_meta")) and getattr(args, name) == -1:
            setattr(args, name, None)
    if args.meta_stepsize == -1:
        args.meta_stepsize = None
    if args.total_steps is None:
        args.total_steps = args.max_iters
    positive = ["batch_size", "gradient_accumulation_steps", "max_seq_len", "vocab_size",
                "dim", "n_layers", "n_heads", "n_kv_heads", "multiple_of", "log_interval", "eval_iters"]
    if any(getattr(args, name) <= 0 for name in positive):
        raise ValueError("Batch/model sizes and logging/evaluation counts must be positive")
    if args.max_iters < 0 or args.num_workers < 0 or args.eval_interval < 0 or args.grad_clip < 0:
        raise ValueError("Iteration/worker counts, evaluation interval and clip must be nonnegative")
    if args.dim % args.n_heads or (args.dim // args.n_heads) % 2 or args.n_heads % args.n_kv_heads:
        raise ValueError("Head dimensions must divide dim, be even for RoPE, and support grouped KV")
    if args.vocab_source == "llama2" and args.vocab_size != 32000:
        raise ValueError("llama2 requires vocab_size=32000; select custom for a custom vocabulary")
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    return args


def seed_everything(seed, rank=0):
    seed = seed + 1000003 * rank
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeBudget:
    def __init__(self, value, clock=time.monotonic):
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("max_time must be hh:mm:ss")
        hours, minutes, seconds = map(int, parts)
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError("Invalid max_time")
        self.clock = clock
        self.started = clock()
        self.seconds = hours * 3600 + minutes * 60 + seconds

    def expired(self):
        expired = self.clock() - self.started >= self.seconds
        if dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
            flag = torch.tensor(int(expired), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            expired = bool(flag.item())
        return expired


class ReplayStream:
    def __init__(self, iterator, consumed=0):
        self.iterator = iter(iterator)
        self.consumed = 0
        for _ in range(consumed):
            next(self)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        self.consumed += 1
        return batch


def accumulated_step(model, raw_model, optimizer, batches, accumulation_steps,
                     scaler, grad_clip, autocast_context=nullcontext, expired=lambda: False):
    """One backward per microbatch, unscale/clip once, one scaler-controlled step.

    Batches have equal token counts. An interrupted accumulation is discarded,
    but its already consumed data position is preserved by ReplayStream.
    """
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for micro_step in range(accumulation_steps):
        if expired():
            optimizer.zero_grad(set_to_none=True)
            return None
        x, y = next(batches)
        if expired():
            optimizer.zero_grad(set_to_none=True)
            return None
        sync = model.no_sync() if isinstance(model, DDP) and micro_step < accumulation_steps - 1 else nullcontext()
        with sync:
            with autocast_context():
                model(x, y)
                loss = raw_model.last_loss
            total_loss += loss.detach().item()
            scaler.scale(loss / accumulation_steps).backward()
    if expired():
        optimizer.zero_grad(set_to_none=True)
        return None
    scaler.unscale_(optimizer)
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer, raw_model, loss=None)
    scaler.update()
    mean_loss = torch.tensor(total_loss / accumulation_steps, device=next(raw_model.parameters()).device)
    if dist.is_initialized():
        dist.all_reduce(mean_loss)
        mean_loss /= dist.get_world_size()
    return mean_loss.item()


@torch.no_grad()
def estimate_loss(model, raw_model, make_batches, eval_iters, autocast_context=nullcontext,
                  expired=lambda: False):
    model.eval()
    losses = {}
    try:
        for split in ("train", "val"):
            batches = iter(make_batches(split))
            values = []
            for _ in range(eval_iters):
                if expired():
                    return None
                x, y = next(batches)
                if expired():
                    return None
                with autocast_context():
                    model(x, y)
                values.append(raw_model.last_loss.item())
            value = torch.tensor(sum(values) / len(values), device=next(raw_model.parameters()).device)
            if dist.is_initialized():
                dist.all_reduce(value)
                value /= dist.get_world_size()
            losses[split] = value.item()
    finally:
        model.train()
    return losses


def log_metrics(optimizer, step, optimizer_updates, train_loss=None, losses=None, recorder=None,
                writer=None, optimizer_step_skipped=None):
    rates = [float(rate) for rate in optimizer.get_learning_rates()]
    metrics = {"iter": step, "optimizer_updates": optimizer_updates,
               "optimizer_step_skips": step - optimizer_updates, "lr": rates[0], "learning_rates": rates}
    if optimizer_step_skipped is not None:
        metrics["optimizer_step_skipped"] = optimizer_step_skipped
    if train_loss is not None:
        metrics["loss/train"] = float(train_loss)
    if losses is not None:
        metrics.update({f"loss/{split}": float(value) for split, value in losses.items()})
    if recorder is not None:
        recorder.log(metrics, step=step)
    if writer is not None:
        for name, value in metrics.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(name, value, step)
    return metrics


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def dataset_identity(args):
    shards = token_shards(args.data_dir, args.vocab_source, args.vocab_size)
    if len(shards) < 2:
        raise ValueError("Need at least two token shards (first validation, remaining training)")
    # Content hashing is deliberately thorough, and can be expensive on full data.
    records = [{"file": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in shards]
    manifest_path = shards[0].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        expected = {r["file"]: r["sha256"] for r in manifest["shards"]}
        if expected != {r["file"]: r["sha256"] for r in records}:
            raise ValueError("Token shards do not match preprocessing manifest")
    tokenizer = Path(args.tokenizer_model or (Path(args.data_dir) / f"tok{args.vocab_size}.model"
                     if args.vocab_source == "custom" else "tokenizer.model"))
    tokenizer_hash = file_sha256(tokenizer) if tokenizer.exists() else None
    if manifest is not None and tokenizer_hash is not None and tokenizer_hash != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match preprocessing manifest")
    return {"shards": records, "split": "first shard val; remaining train",
            "tokenizer": str(tokenizer),
            "tokenizer_sha256": tokenizer_hash or (manifest or {}).get("tokenizer_sha256"),
            "tokenizer_status": "local" if tokenizer_hash else ("preprocessing manifest" if manifest else "unavailable; pretokenized inputs only")}


def resume_signature(config):
    ignored = {"init_from", "checkpoint", "save_directory", "run_name", "max_time", "max_iters",
               "wandb_log", "wandb_project", "tensorboard", "eval_only", "eval_interval", "eval_iters", "log_interval"}
    return {**config, "args": {k: v for k, v in config["args"].items() if k not in ignored}}


def save_checkpoint(run_dir, raw_model, optimizer, scaler, stream, iteration, optimizer_updates,
                    best_val_loss, config, exact_resume=True):
    state = {"format_version": 2, "exact_resume": exact_resume,
             "resume_limitations": "single process, num_workers=0; O(consumed batches) deterministic replay",
             "model": raw_model.state_dict(), "model_args": asdict(raw_model.params),
             "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "rng": rng_state(), "data_batches": stream.consumed, "iter_num": iteration,
             "optimizer_updates": optimizer_updates,
             "best_val_loss": best_val_loss, "config": config}
    temporary = run_dir / "ckpt.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, run_dir / "ckpt.pt")
    from export import model_export
    temporary = run_dir / "model.bin.tmp"
    model_export(raw_model, str(temporary), version=0)
    os.replace(temporary, run_dir / "model.bin")


def run(args, clock=time.monotonic, recorder=None):
    from build_optimizer import build_optimizer
    budget = TimeBudget(args.max_time, clock)
    ddp = int(os.environ.get("RANK", -1)) != -1
    rank = int(os.environ["RANK"]) if ddp else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if ddp else 1
    if args.init_from == "resume" and (ddp or args.num_workers):
        raise ValueError("Exact resume requires a single process and num_workers=0; use explicit warmstart instead")
    device = args.device
    if ddp:
        if device.startswith("cuda"):
            device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.startswith("cuda") else "gloo")
    writer = None
    wandb_run = None
    try:
        seed_everything(args.seed, rank)
        run_dir = Path(args.save_directory) / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "ckpt.pt"
        if args.init_from != "resume" and (run_dir / "ckpt.pt").exists():
            raise ValueError("Run already contains a checkpoint; choose a new run_name or explicit resume")
        checkpoint = None
        if args.init_from != "scratch":
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if args.init_from == "resume" and (
                    checkpoint.get("format_version") != 2 or not checkpoint.get("exact_resume")
                    or type(checkpoint.get("optimizer_updates")) is not int
                    or type(checkpoint.get("iter_num")) is not int
                    or not 0 <= checkpoint["optimizer_updates"] <= checkpoint["iter_num"]):
                raise ValueError("Checkpoint lacks supported exact-resume/update accounting; use explicit --init-from warmstart")
        model_args = {name: getattr(args, name) for name in
                      ("dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len", "dropout", "attention")}
        raw_model = Transformer(ModelArgs(**model_args)).to(device)
        if checkpoint is not None:
            weights = {name.removeprefix("_orig_mod."): value for name, value in checkpoint["model"].items()}
            raw_model.load_state_dict(weights)
        model = torch.compile(raw_model) if args.compile else raw_model
        if rank == 0:
            print(f"compile={args.compile}; attention={args.attention} (SDPA chooses its own device kernel)")
        if ddp:
            model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])] if device.startswith("cuda") else None)
        # Construct meta state only after DDP has broadcast rank zero's model.
        if args.tensorboard and rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(run_dir / "tensorboard"))
        optimizer = build_optimizer(raw_model, args, writer)
        scaler = torch.amp.GradScaler(torch.device(device).type, enabled=args.dtype == "float16")
        precision = getattr(torch, args.dtype)
        def autocast_context():
            return nullcontext() if args.dtype == "float32" else torch.autocast(torch.device(device).type, dtype=precision)
        def make_batches(split):
            return Task.iter_batches(batch_size=args.batch_size, device=device, num_workers=args.num_workers,
                                     split=split, max_seq_len=args.max_seq_len, vocab_size=args.vocab_size,
                                     vocab_source=args.vocab_source, seed=args.seed, data_dir=args.data_dir, rank=rank)
        config = {"args": vars(args).copy(), "model_args": asdict(raw_model.params),
                  "optimizer": optimizer.config, "dataset": dataset_identity(args),
                  "global_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                  "tokens_per_iteration": args.batch_size * args.gradient_accumulation_steps * world_size * args.max_seq_len,
                  "precision": args.dtype, "compiled": args.compile, "attention": args.attention,
                  "world_size": world_size, "device": device,
                  "exact_resume_supported": not ddp and args.num_workers == 0}
        iteration, optimizer_updates, best_val_loss = 0, 0, float("inf")
        consumed = 0
        if args.init_from == "resume":
            if resume_signature(checkpoint["config"]) != resume_signature(config):
                raise ValueError("Effective configuration differs from checkpoint; use warmstart for changed runs")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            iteration, best_val_loss = checkpoint["iter_num"], checkpoint["best_val_loss"]
            optimizer_updates = checkpoint["optimizer_updates"]
            consumed = checkpoint["data_batches"]
        original_step = optimizer.step
        def counted_step(*step_args, **step_kwargs):
            nonlocal optimizer_updates
            result = original_step(*step_args, **step_kwargs)
            optimizer_updates += 1
            return result
        # GradScaler calls this only when it actually executes the optimizer step.
        optimizer.step = counted_step
        stream = ReplayStream(make_batches("train"), consumed)
        if args.init_from == "resume":
            restore_rng(checkpoint["rng"])
        if rank == 0:
            temporary = run_dir / "config.json.tmp"
            temporary.write_text(json.dumps(config, indent=2) + "\n")
            os.replace(temporary, run_dir / "config.json")
            if args.wandb_log and recorder is None:
                import wandb
                wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=config, dir=str(run_dir))
                recorder = wandb_run
        while iteration < args.max_iters and not budget.expired():
            if args.eval_only or (args.eval_interval and iteration % args.eval_interval == 0):
                losses = estimate_loss(model, raw_model, make_batches, args.eval_iters, autocast_context, budget.expired)
                if losses is None:
                    break
                best_val_loss = min(best_val_loss, losses["val"])
                if rank == 0:
                    log_metrics(optimizer, iteration, optimizer_updates, losses=losses,
                                recorder=recorder, writer=writer)
                if args.eval_only:
                    break
            if budget.expired():
                break
            previous_updates = optimizer_updates
            loss = accumulated_step(model, raw_model, optimizer, stream, args.gradient_accumulation_steps,
                                    scaler, args.grad_clip, autocast_context, budget.expired)
            if loss is None:
                break
            optimizer_step_skipped = optimizer_updates == previous_updates
            iteration += 1
            if rank == 0 and iteration % args.log_interval == 0:
                metrics = log_metrics(optimizer, iteration, optimizer_updates, train_loss=loss,
                                      recorder=recorder, writer=writer,
                                      optimizer_step_skipped=optimizer_step_skipped)
                print(f"attempt {iteration}: optimizer_updates={optimizer_updates}, "
                      f"optimizer_step_skipped={optimizer_step_skipped}, loss {loss:.4f}, lr {metrics['lr']:.6g}")
            if args.eval_interval and iteration % args.eval_interval == 0 and rank == 0:
                save_checkpoint(run_dir, raw_model, optimizer, scaler, stream, iteration, optimizer_updates,
                                best_val_loss, config, config["exact_resume_supported"])
        if rank == 0:
            save_checkpoint(run_dir, raw_model, optimizer, scaler, stream, iteration, optimizer_updates,
                            best_val_loss, config, config["exact_resume_supported"])
        return {"iter_num": iteration, "optimizer_updates": optimizer_updates,
                "optimizer_step_skips": iteration - optimizer_updates,
                "data_batches": stream.consumed, "config": config}
    finally:
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
