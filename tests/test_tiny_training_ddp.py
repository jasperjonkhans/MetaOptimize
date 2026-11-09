"""Two-process CPU/Gloo coverage; deliberately does not claim NCCL validation."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


# Check in the parent process before launching workers that import tinystories.py.
for dependency in ("requests", "tqdm", "sentencepiece"):
    pytest.importorskip(dependency, exc_type=ModuleNotFoundError)


@pytest.mark.parametrize("optimizer_name", ["HF", "AdamW"])
def test_ddp_final_microbatch_syncs_parameters_and_optimizer_state(tmp_path, optimizer_name):
    source = Path(__file__).resolve().parents[1] / "codes/MetaOptimize/tinystories"
    script = tmp_path / "ddp_check.py"
    script.write_text(textwrap.dedent('''
        import copy
        import os
        import sys
        sys.path.insert(0, sys.argv[1])
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        import train
        from build_optimizer import build_optimizer

        def equal(a, b, exact=True):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0 if exact else 3e-5, atol=0 if exact else 2e-7)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a: equal(a[k], b[k], exact)
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for x, y in zip(a, b): equal(x, y, exact)
            else:
                assert a == b

        dist.init_process_group('gloo')
        rank = dist.get_rank()
        # Distinct initial RNGs force this test to exercise DDP's broadcast.
        train.seed_everything(17, rank)
        model = train.Transformer(train.ModelArgs(dim=8, n_layers=1, n_heads=2,
                    n_kv_heads=2, vocab_size=32, multiple_of=4, max_seq_len=4))
        ddp = DDP(model)
        reference = copy.deepcopy(model)
        args = train.parse_args(['--optimizer', sys.argv[2], '--alpha0', '.002',
                                '--warmup-steps', '0', '--total-steps', '4'])
        optimizer = build_optimizer(model, args, None)
        ref_optimizer = build_optimizer(reference, args, None)
        x = torch.arange(16).reshape(4, 4)
        y = (x + 3) % 32
        local_x, local_y = x[2*rank:2*rank+2], y[2*rank:2*rank+2]
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        for _ in range(2):
            loss = train.accumulated_step(ddp, model, optimizer,
                      iter(zip(local_x.chunk(2), local_y.chunk(2))), 2, scaler, .01)
            expected_loss = train.accumulated_step(reference, reference, ref_optimizer,
                      iter([(x, y)]), 1, scaler, .01)
            assert abs(loss - expected_loss) < 1e-6
            equal(model.state_dict(), reference.state_dict(), exact=False)
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, (model.state_dict(), optimizer.state_dict()))
            equal(states[0], states[1])
        if rank == 0: print('DDP parameters and all optimizer states agree after two updates')
        dist.destroy_process_group()
    '''))
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(script), str(source), optimizer_name,
    ], env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
       capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (result.stdout + result.stderr)[-8000:]
    assert "all optimizer states agree" in result.stdout


def test_ddp_run_final_checkpoint_records_actual_global_batch(tmp_path):
    import json
    import numpy as np
    import torch

    source = Path(__file__).resolve().parents[1] / "codes/MetaOptimize/tinystories"
    data = tmp_path / "data/tok32"
    data.mkdir(parents=True)
    for i in range(3):
        ((np.arange(97, dtype=np.uint16) + i * 7) % 32).tofile(data / f"shard{i}.bin")
    script = tmp_path / "ddp_run.py"
    script.write_text("import sys\nsys.path.insert(0, sys.argv.pop(1))\nimport train\ntrain.run(train.parse_args())\n")
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(script), str(source), "--device", "cpu", "--optimizer", "HF",
        "--vocab-source", "custom", "--vocab-size", "32", "--dim", "8", "--n-layers", "1",
        "--n-heads", "2", "--n-kv-heads", "2", "--multiple-of", "4", "--max-seq-len", "4",
        "--batch-size", "2", "--gradient-accumulation-steps", "2", "--max-iters", "2",
        "--warmup-steps", "0", "--total-steps", "4", "--eval-interval", "1", "--eval-iters", "1",
        "--data-dir", str(tmp_path / "data"), "--save-directory", str(tmp_path / "runs"),
        "--run-name", "ddp",
    ], env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
       capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (result.stdout + result.stderr)[-8000:]
    run_dir = tmp_path / "runs/ddp"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["world_size"] == 2 and config["global_batch_size"] == 8
    assert config["tokens_per_iteration"] == 32 and not config["exact_resume_supported"]
    state = torch.load(run_dir / "ckpt.pt", weights_only=False)
    assert state["iter_num"] == 2 and state["data_batches"] == 4
    assert state["format_version"] == 2 and state["optimizer_updates"] == 2
    assert not state["exact_resume"]
    assert (run_dir / "model.bin").stat().st_size > 0
