"""Atomic state-load regressions for HF tensors and native scheduler rollback."""
import copy
import io

import pytest
import torch

from test_experiment_optimizers import LANES, hf, load


def serialized(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def optimizer_pair(lane, base='AdamW', meta='Adam', native=False, dtype=torch.double):
    model = torch.nn.Linear(2, 1).to(dtype=dtype)

    def build(net):
        if native:
            return load(lane, 'build_optimizer').AdamW_optimizer(
                net, .1, warmup_steps=0, total_steps=4)
        return hf(lane, net, base=base, meta=meta, groups='layerwise')

    optimizer = build(model)
    for value in [1., -2.]:
        optimizer.step(model, model(torch.full((1, 2), value, dtype=dtype)).sum())
    reference_model = copy.deepcopy(model)
    reference = build(reference_model)
    reference.load_state_dict(optimizer.state_dict())
    return model, optimizer, reference_model, reference


def assert_rejected_load_is_atomic_and_continues(pair, malformed):
    model, optimizer, reference_model, reference = pair
    before_state = serialized(optimizer.state_dict())
    before_weights = serialized(model.state_dict())
    before_rates = optimizer.get_learning_rates()
    with pytest.raises(ValueError):
        optimizer.load_state_dict(malformed)
    assert serialized(optimizer.state_dict()) == before_state
    assert serialized(model.state_dict()) == before_weights
    assert optimizer.get_learning_rates() == before_rates
    if hasattr(optimizer, 'optimizer'):
        assert optimizer.param_groups is optimizer.optimizer.param_groups
    for value in [3., -1.]:
        x = torch.full((1, 2), value, dtype=model.weight.dtype)
        optimizer.step(model, model(x).square().mean())
        reference.step(reference_model, reference_model(x).square().mean())
        assert serialized(optimizer.state_dict()) == serialized(reference.state_dict())
        assert serialized(model.state_dict()) == serialized(reference_model.state_dict())
        assert optimizer.get_learning_rates() == reference.get_learning_rates()
        if hasattr(optimizer, 'optimizer'):
            assert optimizer.param_groups is optimizer.optimizer.param_groups
            assert optimizer.get_learning_rates() == [g['lr'] for g in optimizer.optimizer.param_groups]


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('field', ['beta', 'h_condenced', 'trace_base', 'momentum_base',
                                   'trace_meta', 'momentum_meta'])
@pytest.mark.parametrize('corruption', ['sparse', 'integer', 'nan', 'inf', 'shape', 'conversion_overflow'])
def test_hf_malformed_tensor_load_is_atomic(lane, field, corruption):
    pair = optimizer_pair(lane, dtype=torch.float32 if corruption == 'conversion_overflow' else torch.double)
    malformed = pair[1].state_dict()
    value = malformed['state'][field][0]
    if corruption == 'sparse':
        value = value.to_sparse()
    elif corruption == 'integer':
        value = value.long()
    elif corruption == 'shape':
        value = value.reshape(-1).repeat(3)
    elif corruption == 'conversion_overflow':
        # Finite source values must also remain finite in the live model dtype.
        value = torch.full_like(value, 1e100, dtype=torch.double)
    else:
        value.fill_(float(corruption))
    malformed['state'][field][0] = value
    assert_rejected_load_is_atomic_and_continues(pair, malformed)


@pytest.mark.parametrize('lane', LANES)
def test_hf_nonfinite_derived_learning_rate_is_rejected_before_commit(lane):
    pair = optimizer_pair(lane)
    malformed = pair[1].state_dict()
    malformed['state']['beta'][0].fill_(1000.)
    assert_rejected_load_is_atomic_and_continues(pair, malformed)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('field,base,meta', [('trace_base', 'RMSProp', 'fixed'),
                                            ('trace_base', 'AdamW', 'fixed'),
                                            ('trace_meta', 'SGD', 'RMSProp'),
                                            ('trace_meta', 'SGD', 'Adam')])
def test_hf_rejects_negative_second_moments_without_mutation(lane, field, base, meta):
    pair = optimizer_pair(lane, base, meta)
    malformed = pair[1].state_dict()
    malformed['state'][field][0].fill_(-100.)
    assert_rejected_load_is_atomic_and_continues(pair, malformed)


@pytest.mark.parametrize('lane', LANES)
def test_hf_signed_sensitivity_and_first_moments_are_valid(lane):
    model, optimizer, reference_model, reference = optimizer_pair(lane)
    state = optimizer.state_dict()
    for field in ['beta', 'h_condenced', 'momentum_base', 'momentum_meta']:
        for value in state['state'][field]:
            value.fill_(-.25)
    optimizer.load_state_dict(state)
    reference.load_state_dict(state)
    assert serialized(optimizer.state_dict()) == serialized(reference.state_dict())
    x = torch.ones(1, 2, dtype=torch.double)
    optimizer.step(model, model(x).sum())
    reference.step(reference_model, reference_model(x).sum())
    assert serialized(optimizer.state_dict()) == serialized(reference.state_dict())
    assert serialized(model.state_dict()) == serialized(reference_model.state_dict())


@pytest.mark.parametrize('lane', LANES)
def test_native_bad_scheduler_rollback_keeps_live_groups_and_continuation(lane):
    pair = optimizer_pair(lane, native=True)
    malformed = pair[1].state_dict()
    malformed['scheduler'] = 0
    assert_rejected_load_is_atomic_and_continues(pair, malformed)


@pytest.mark.parametrize('lane', LANES)
def test_native_rollback_groups_remain_usable_by_gradscaler(lane):
    model, optimizer, reference_model, reference = optimizer_pair(lane, native=True)
    malformed = optimizer.state_dict()
    malformed['scheduler'] = 0
    with pytest.raises(ValueError):
        optimizer.load_state_dict(malformed)
    assert optimizer.param_groups is optimizer.optimizer.param_groups
    scaler = torch.amp.GradScaler('cpu')
    optimizer.zero_grad()
    reference.zero_grad()
    x = torch.ones(1, 2, dtype=torch.double)
    scaler.scale(model(x).sum()).backward()
    reference_model(x).sum().backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer, model)
    scaler.update()
    reference.step(reference_model)
    assert serialized(optimizer.state_dict()) == serialized(reference.state_dict())
    assert serialized(model.state_dict()) == serialized(reference_model.state_dict())
