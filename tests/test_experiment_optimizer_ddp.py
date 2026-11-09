"""Two real Gloo processes; each HF state must follow the global-batch reference."""
import copy
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_experiment_optimizers import hf, equal, LANES


def run_rank(rank, init_path, lane):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=2)
    try:
        for meta in ['fixed', 'RMSProp', 'Adam', 'Lion']:
            for base in ['SGD', 'SGDm', 'RMSProp', 'AdamW', 'Lion']:
                local = torch.nn.Linear(2, 1, bias=True).double()
                with torch.no_grad():
                    local.weight.fill_(1.)
                    local.bias.fill_(0.)
                reference = copy.deepcopy(local)
                distributed = torch.nn.parallel.DistributedDataParallel(local)
                optimizer = hf(lane, distributed, base, meta, 'layerwise')
                global_optimizer = hf(lane, reference, base, meta, 'layerwise')
                for step in range(3):
                    # Different local gradient signs/magnitudes, with nonzero mean.
                    xs = [torch.tensor([[1. + step, -1.]], dtype=torch.double),
                          torch.tensor([[-2., 3. + step]], dtype=torch.double)]
                    if step % 2 == 0:
                        optimizer.step(distributed, distributed(xs[rank]).square().mean())
                    else:
                        optimizer.zero_grad()
                        # Accumulation with a final synchronized backward.
                        with distributed.no_sync():
                            (distributed(xs[rank]).square().mean() / 2).backward()
                        (distributed(xs[rank]).square().mean() / 2).backward()
                        optimizer.step(distributed)
                    global_optimizer.step(reference, reference(torch.cat(xs)).square().mean())
                    for actual, expected in zip(local.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    # Exact inter-rank equality for weights AND all HF meta/base state.
                    payload = [p.detach().clone() for p in local.parameters()], optimizer.state_dict()
                    gathered = [None, None]
                    dist.all_gather_object(gathered, payload)
                    equal(gathered[0], gathered[1])
                    actual_state = optimizer.state_dict()['state']
                    expected_state = global_optimizer.state_dict()['state']
                    for key in optimizer._state_fields:
                        if isinstance(actual_state[key], list):
                            for actual, expected in zip(actual_state[key], expected_state[key]):
                                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                        else:
                            assert actual_state[key] == expected_state[key]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('lane', LANES)
def test_two_cpu_gloo_hf_matches_global_batch(lane):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(run_rank, args=(str(Path(directory) / 'rendezvous'), lane), nprocs=2, join=True)
