"""HF bias-correction checkpoint products stay in their reachable [0, 1] domain."""
import pytest
import torch

from test_experiment_optimizers import LANES
from test_experiment_optimizer_state_load import (
    assert_rejected_load_is_atomic_and_continues, optimizer_pair, serialized)


CLOCK_ALGORITHMS = [('lambda_base_t', 'AdamW', 'fixed'),
                    ('lambda_base_t', 'RMSProp', 'fixed'),
                    ('lambda_meta_t', 'SGD', 'RMSProp'),
                    ('lambda_meta_t', 'SGD', 'Adam')]


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('clock,base,meta', CLOCK_ALGORITHMS)
@pytest.mark.parametrize('value', [-.1, 2.])
def test_hf_out_of_domain_clock_load_is_atomic(lane, clock, base, meta, value):
    pair = optimizer_pair(lane, base, meta)
    malformed = pair[1].state_dict()
    malformed['state'][clock] = value
    input_before = serialized(malformed)
    assert_rejected_load_is_atomic_and_continues(pair, malformed)
    assert serialized(malformed) == input_before


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('clock,base,meta', CLOCK_ALGORITHMS)
@pytest.mark.parametrize('value', [0., 1.])
def test_hf_boundary_clock_load_and_continuation(lane, clock, base, meta, value):
    model, optimizer, reference_model, reference = optimizer_pair(lane, base, meta)
    state = optimizer.state_dict()
    state['state'][clock] = value
    input_before = serialized(state)
    optimizer.load_state_dict(state)
    reference.load_state_dict(state)
    assert optimizer.state_dict()['state'][clock] == value
    for gradient in [1., -2.]:
        x = torch.full((1, 2), gradient, dtype=model.weight.dtype)
        optimizer.step(model, model(x).sum())
        reference.step(reference_model, reference_model(x).sum())
        assert serialized(optimizer.state_dict()) == serialized(reference.state_dict())
        assert serialized(model.state_dict()) == serialized(reference_model.state_dict())
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert serialized(state) == input_before
