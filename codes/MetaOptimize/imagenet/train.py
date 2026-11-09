"""ImageNet training; exact resume supports single-process completed epochs only."""
import argparse
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_runtime import (Budget, dataset_info, epoch_pass, load_weights,
                            make_loader, prepare_run, run_training,
                            seed_everything, time_seconds, write_json)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='/datashare/ImageNet/ILSVRC2012/',
                    help='path to dataset (default: imagenet)')
parser.add_argument("--save-directory", type=str, default='data',
                        help="parent directory for isolated run outputs")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--optimizer', metavar='OPT', default='HF', choices=['SGD', 'SGDm', 'Adam', 'AdamW', 'Lion', 'HF'],
                    help='The optimizer algorithm')
parser.add_argument("--run-name", type=str, default="test",
                        help="name of the current run")
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--base-lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='base_lr')
parser.add_argument('--momentum', dest='base_momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
restart = parser.add_mutually_exclusive_group()
restart.add_argument('--resume', default='', type=str, metavar='PATH',
                     help='full single-process completed-epoch checkpoint (fresh run-name required)')
restart.add_argument('--warmstart', default='', type=str, metavar='PATH',
                     help='weights only, including legacy/partial/distributed checkpoints')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', default=False, action='store_true', help="use fake data to benchmark")



# Arguemnts for Base and Meta algorithms
parser.add_argument("--alg-base", type=str, default="AdamW")
parser.add_argument("--normalizer-param-base", type=float, default=0.999)
parser.add_argument("--momentum-param-base", type=float, default=0.9)
parser.add_argument("--weight-decay-base", "--wd", "--weight-decay", type=float, default=0.1)
parser.add_argument("--Lion-beta2-base", type=float, default=-1)

parser.add_argument("--alg-meta", type=str, default="Lion")
parser.add_argument("--normalizer-param-meta", type=float, default=-1)
parser.add_argument("--momentum-param-meta", type=float, default=0.99)
parser.add_argument("--weight-decay-meta", type=float, default=0)
parser.add_argument("--Lion-beta2-meta", type=float, default=0.9)

parser.add_argument("--stepsize-groups", type=str, default="scalar")



parser.add_argument("--meta-stepsize", type=float, default=1e-3)
parser.add_argument("--alpha0", type=float, default=1e-5)
parser.add_argument("--gamma", type=float, default=1)
    



parser.add_argument("--max-time", type=str, default="24:00:00",
                        help="maximum runtime of the program in hh:mm:ss format")

parser.add_argument("--deterministic", action="store_true",
                    help="deterministic kernels (may be slower or reject unsupported operations)")
parser.add_argument("--warmup-steps", type=int, default=0)
parser.add_argument("--total-steps", type=int, default=None,
                    help="schedule length; defaults to epochs * batches per epoch")


def main(argv=None):
    args = parser.parse_args(argv)
    for name in ("normalizer_param", "momentum_param", "weight_decay", "Lion_beta2"):
        for level in ("base", "meta"):
            key = f"{name}_{level}"
            if getattr(args, key) == -1:
                setattr(args, key, None)
    if args.meta_stepsize == -1:
        args.meta_stepsize = None
    if args.epochs < 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/workers must be nonnegative and batch-size positive")
    if args.start_epoch:
        raise ValueError("--start-epoch alone is not a resume; use --resume or --warmstart")
    if args.evaluate and args.resume:
        raise ValueError("--evaluate accepts --warmstart weights, not a training --resume")
    if args.dist_url == "env://":
        args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.rank = int(os.environ.get("RANK", args.rank))
    if args.multiprocessing_distributed and args.world_size == -1:
        args.world_size = 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed and args.resume:
        raise ValueError("distributed --resume is unsupported; use --warmstart (not exact replay)")
    # Include setup and spawning in the per-invocation elapsed budget.
    import time
    args.invocation_start = time.monotonic()
    time_seconds(args.max_time)
    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.multiprocessing_distributed:
        if not torch.cuda.is_available():
            raise ValueError("multiprocessing-distributed requires CUDA; use torchrun + gloo for CPU DDP")
        args.world_size *= ngpus
        args.rank = max(0, args.rank)
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
    else:
        if args.distributed and args.gpu is None and torch.cuda.is_available():
            args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        main_worker(args.gpu, ngpus, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    seed_everything(args.seed, args.deterministic)
    if torch.cuda.is_available():
        if gpu is not None:
            torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu) if gpu is not None else torch.device("cuda")
    elif torch.backends.mps.is_available() and not args.distributed:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
            args.batch_size //= ngpus_per_node
            args.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
        if args.batch_size < 1:
            raise ValueError("batch-size must provide at least one sample per process")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    writer = None
    try:
        primary = not args.distributed or dist.get_rank() == 0
        # Broadcast directory errors as well: peers must not hang at a barrier.
        location = [None, None]
        if primary:
            try:
                location[0] = str(prepare_run(args.save_directory, args.run_name))
            except (ValueError, OSError) as error:
                location[1] = str(error)
        if args.distributed:
            dist.broadcast_object_list(location, src=0, device=device)
        if location[1]:
            raise ValueError(location[1])
        run_dir = Path(location[0])
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tensorboard")) if primary else None
        model = models.__dict__[args.arch](weights="DEFAULT" if args.pretrained else None).to(device)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu] if device.type == "cuda" else None)
        elif device.type == "cuda" and gpu is None and ngpus_per_node > 1:
            model = torch.nn.DataParallel(model)

        train_dataset, val_dataset = load_datasets(args)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, seed=args.seed)
            # Equal-length validation shards avoid mismatched budget collectives.
            # Padding, when necessary, is explicitly recorded in the manifest.
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, seed=args.seed)
            args.validation_sampler = "DistributedSampler with padding to equal shard lengths"
        else:
            train_sampler = val_sampler = None
        train_loader = make_loader(train_dataset, args.batch_size, args.seed,
                                   args.workers, shuffle=True, sampler=train_sampler)
        val_loader = make_loader(val_dataset, args.batch_size, args.seed + 1,
                                 args.workers, sampler=val_sampler)
        budget = Budget(time_seconds(args.max_time), distributed=args.distributed, device=device)
        budget.start = args.invocation_start
        if args.evaluate:
            if args.warmstart:
                load_weights(args.warmstart, model, device)
            metrics = epoch_pass(val_loader, model, torch.nn.CrossEntropyLoss(), device, budget=budget)
            metrics.update(elapsed_seconds=budget.elapsed,
                           stop_reason="evaluation_complete" if metrics["complete"] else "budget")
            if primary:
                write_json(run_dir / "evaluation.json", metrics)
                write_json(run_dir / "manifest.json", {
                    "args": vars(args), "architecture": args.arch, "device": str(device),
                    "precision": "float32", "dataset": dataset_info(val_dataset),
                    "optimizer": None, "scheduler": None})
            return metrics
        if args.total_steps is None:
            args.total_steps = max(1, args.epochs * len(train_loader))
        from Optimizers.build_optimizer import build_optimizer
        optimizer = build_optimizer(model, args, writer)
        return run_training(args, model, optimizer, train_loader, val_loader,
                            writer, run_dir, budget, args.arch, args.epochs, device)
    finally:
        if writer is not None:
            writer.close()
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()


def load_datasets(args):
    if args.dummy:
        return (datasets.FakeData(100000, (3, 224, 224), 1000, transforms.ToTensor()),
                datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor()))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train = datasets.ImageFolder(os.path.join(args.data, "train"), transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize]))
    validation = datasets.ImageFolder(os.path.join(args.data, "validation"), transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
    return train, validation


if __name__ == "__main__":
    main()
