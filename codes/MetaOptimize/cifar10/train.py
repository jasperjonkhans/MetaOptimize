"""CIFAR/MNIST training with isolated outputs and completed-epoch resume."""
import argparse
from pathlib import Path
import sys

import torch

# Also permit launching this file directly from its experiment directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_runtime import (Budget, make_loader, prepare_run, run_training,
                            seed_everything, time_seconds)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--save-directory", type=str, default='outputs',
                        help="parent directory for isolated run outputs")
    parser.add_argument("--run-name", type=str, default="test",
                        help="name of the current run")
    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        help="name of the current run")
    parser.add_argument("--NN-name", type=str, default='ResNet18', # M1 is a 2-layer small network, M2 is a 3-layer network of width 512 for MNIST
                        help="name of Neural Network")
    parser.add_argument("--verbos", type=int, default=1,
                        help="how much printing. 0:minimal printing,  1:moderate_printing")
    parser.add_argument("--max-time", type=str, default="00:05:00",
                        help="maximum runtime of the program in hh:mm:ss format")
    
    
    parser.add_argument("--deterministic", action="store_true",
                        help="use deterministic algorithms (may be slower or reject unsupported kernels)")
    parser.add_argument("--workers", type=int, default=0)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", default="", help="full completed-epoch checkpoint; choose a fresh run-name")
    restart.add_argument("--warmstart", default="", help="weights only, including legacy/partial checkpoints")
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--momentum", dest="base_momentum", type=float, default=0.9)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=None,
                        help="schedule length; defaults to num-epochs * batches per epoch")

    # Trainig
    parser.add_argument("--optimizer", type=str, default='HF', choices=['AdamW', 'HF'],
                        help="name of the base and meta optimizer (if we have a meta optimizer)")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="samples per training batch")
    
    
     
    # Arguemnts for Base and Meta algorithms
    parser.add_argument("--alg-base", type=str, default="AdamW")
    parser.add_argument("--normalizer-param-base", type=float, default=0.999)
    parser.add_argument("--momentum-param-base", type=float, default=0.9)
    parser.add_argument("--weight-decay-base", type=float, default=0.1)
    parser.add_argument("--Lion-beta2-base", type=float, default=-1)

    parser.add_argument("--alg-meta", type=str, default="Lion")
    parser.add_argument("--normalizer-param-meta", type=float, default=-1)
    parser.add_argument("--momentum-param-meta", type=float, default=0.99)
    parser.add_argument("--weight-decay-meta", type=float, default=0)
    parser.add_argument("--Lion-beta2-meta", type=float, default=0.9)

    parser.add_argument("--stepsize-groups", type=str, default="scalar")
    parser.add_argument("--gamma", type=float, default=1, help="decay factor")
    parser.add_argument("--meta-stepsize", type=float, default=10e-3)
    parser.add_argument("--alpha0", type=float, default=1e-6)

    args = parser.parse_args(argv)
    return args





def main(argv=None):
    args = parse_args(argv)
    for name in ("normalizer_param", "momentum_param", "weight_decay", "Lion_beta2"):
        for level in ("base", "meta"):
            key = f"{name}_{level}"
            if getattr(args, key) == -1:
                setattr(args, key, None)
    if args.meta_stepsize == -1:
        args.meta_stepsize = None
    if args.num_epochs < 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/workers must be nonnegative and batch-size positive")
    budget = Budget(time_seconds(args.max_time))
    seed_everything(args.seed, args.deterministic)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = prepare_run(args.save_directory, args.run_name)
    # Optional experiment dependencies are imported only when actually training.
    from torch.utils.tensorboard import SummaryWriter
    from build_network import build_network
    from load_data import load_data
    from Optimizers.build_optimizer import build_optimizer

    writer = SummaryWriter(str(run_dir / "tensorboard"))
    try:
        net = build_network(args.NN_name, args.device)
        original_train, original_test = load_data(args.dataset, args.batch_size, args.seed)
        train_loader = make_loader(original_train.dataset, args.batch_size, args.seed,
                                   args.workers, shuffle=True)
        test_loader = make_loader(original_test.dataset, args.batch_size, args.seed + 1,
                                  args.workers)
        if args.total_steps is None:
            args.total_steps = max(1, args.num_epochs * len(train_loader))
        optimizer = build_optimizer(net, args, writer)
        return run_training(args, net, optimizer, train_loader, test_loader, writer,
                            run_dir, budget, args.NN_name, args.num_epochs, args.device)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
