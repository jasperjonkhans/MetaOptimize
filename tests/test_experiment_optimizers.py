"""Real-framework regressions for the three standalone experiment optimizers."""
import copy
import importlib.util
import json
import math
from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'codes' / 'MetaOptimize'
LANES = ['cifar10', 'imagenet', 'tinystories']


def load(lane, module):
    directory = ROOT / lane
    if lane != 'tinystories':
        directory /= 'Optimizers'
    name = f'experiment_{lane}_{module}'
    spec = importlib.util.spec_from_file_location(name, directory / f'{module}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    if module == 'HF':
        sys.modules['HF' if lane == 'tinystories' else 'Optimizers.HF'] = result
    return result


def args(optimizer='HF', **overrides):
    values = dict(optimizer=optimizer, alpha0=0.1, lr=0.03, momentum=0.8,
                  weight_decay=0.02, alg_base='AdamW', normalizer_param_base=0.99,
                  momentum_param_base=0.9, weight_decay_base=0.01, Lion_beta2_base=0.9,
                  alg_meta='Lion', normalizer_param_meta=-1, momentum_param_meta=0.9,
                  weight_decay_meta=0., Lion_beta2_meta=0.99, meta_stepsize=0.001,
                  stepsize_groups='scalar', gamma=1., max_iters=20)
    values.update(overrides)
    return Namespace(**values)


def hf(lane, model, base='AdamW', meta='Lion', groups='scalar', **extra):
    return load(lane, 'HF').HF(model, groups, extra.pop('alpha0', 0.1),
                              dict(alg=base, weight_decay=0., momentum_param=0.9,
                                   normalizer_param=0.99, Lion_beta2=0.99),
                              dict(alg=meta, weight_decay=0., momentum_param=0.9,
                                   normalizer_param=0.99, Lion_beta2=0.99, meta_stepsize=0.01),
                              gamma=1., **extra)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('choice', ['HF', 'AdamW', 'SGD', 'SGDm', 'Adam', 'Lion'])
def test_offered_builders_real_updates(lane, choice):
    if lane != 'imagenet' and choice not in ('HF', 'AdamW'):
        pytest.skip('not offered by this experiment')
    load(lane, 'HF')
    model = torch.nn.Linear(2, 1)
    optimizer = load(lane, 'build_optimizer').build_optimizer(model, args(choice), None)
    json.dumps(optimizer.config)
    before = model.weight.detach().clone()
    optimizer.zero_grad()
    optimizer.step(model, model(torch.ones(2, 2)).square().mean())
    optimizer.zero_grad(set_to_none=False)
    model(torch.ones(2, 2)).square().mean().backward()
    optimizer.step(model)
    assert not torch.equal(before, model.weight)
    assert all(type(rate) is float for rate in optimizer.get_learning_rates())
    assert sum(len(g['params']) for g in optimizer.param_groups) == len(list(model.parameters()))
    if choice in ('SGD', 'SGDm', 'Adam'):
        assert optimizer.config['lr'] == 0.03
        assert optimizer.config['weight_decay'] == 0.02
    if choice == 'AdamW':
        assert optimizer.config['weight_decay'] == 0.01
        assert optimizer.config['betas'] == [0.9, 0.99]
        assert optimizer.config['total_steps'] == (20 if lane == 'tinystories' else None)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('groups', ['scalar', 'layerwise', '[1, 1]', [2], [['weight'], ['bias']]])
def test_supported_groups_two_updates(lane, groups):
    model = torch.nn.Linear(1, 1)
    optimizer = hf(lane, model, groups=groups)
    for _ in range(2):
        optimizer.step(model, model(torch.ones(1, 1)).sum())
    assert len(optimizer.get_learning_rates()) == (1 if groups in ('scalar', [2]) else 2)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('name,sizes', [('resnet18_blocks', [3, 12, 15, 15, 15, 2]),
                                       ('resnet50_blocks', [3, 30, 39, 57, 30, 2])])
def test_named_resnet_groups(lane, name, sizes):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(())) for _ in range(sum(sizes))])
    optimizer = hf(lane, model, groups=name)
    for _ in range(2):
        optimizer.step(model, sum(model))
    assert [len(g['params']) for g in optimizer.param_groups] == sizes


def test_tinystories_named_transformer_groups():
    sizes = [1, 9, 9, 9, 9, 9, 10]
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(())) for _ in range(sum(sizes))])
    optimizer = hf('tinystories', model, groups='transformer_blocks_15m')
    for _ in range(2):
        optimizer.step(model, sum(model))
    assert [len(g['params']) for g in optimizer.param_groups] == sizes
    with pytest.raises(ValueError, match='sum'):
        hf('tinystories', torch.nn.Linear(1, 1), groups='transformer_blocks_15m')


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('groups', ['nodewise', 'weightwise', 'unknown', [], [1], [0, 2], [-1, 3],
                                   [0.5, 1.5], [True, 1], [['weight']],
                                   [['weight'], ['weight', 'bias']], [['weight'], ['nope']]])
def test_invalid_groups_fail_early(lane, groups):
    with pytest.raises(ValueError):
        hf(lane, torch.nn.Linear(1, 1), groups=groups)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('base', ['SGD', 'SGDm', 'RMSProp', 'AdamW', 'Lion'])
@pytest.mark.parametrize('meta', ['fixed', 'RMSProp', 'Adam', 'Lion'])
def test_loss_and_supplied_gradients_agree(lane, base, meta):
    a = torch.nn.Linear(2, 1)
    b = copy.deepcopy(a)
    first, second = hf(lane, a, base, meta), hf(lane, b, base, meta)
    for value in [1., -2.]:
        x = torch.full((2, 2), value)
        first.step(a, a(x).square().mean())
        second.zero_grad()
        # Two backwards are accumulated, not recomputed or cleared by step.
        (b(x).square().mean() / 2).backward()
        (b(x).square().mean() / 2).backward()
        second.step(b)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)
    # Small sum-order differences from accumulation are expected in floating point.
    for key in ['beta', 'trace_meta', 'momentum_meta', 'h_condenced']:
        for va, vb in zip(first.state_dict()['state'][key], second.state_dict()['state'][key]):
            torch.testing.assert_close(va, vb)


@pytest.mark.parametrize('lane', LANES)
def test_historical_adamw_hand_calculation(lane):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((), dtype=torch.float64))])
    optimizer = hf(lane, model, meta='fixed')
    for expected in [0.9, 0.71]:
        optimizer.step(model, model[0])
        assert model[0].item() == pytest.approx(expected, abs=1e-10)
    optimizer.step(model, -2 * model[0])
    variance = (0.01 / (1 - 0.99**3)) * (0.99**2 + 0.99 + 4)
    expected = 0.71 - 0.1 * (0.9 * 1.9 - 2) / math.sqrt(variance + 1e-10)
    assert model[0].item() == pytest.approx(expected, abs=1e-10)
    assert optimizer.h_condenced[0].item() == pytest.approx(model[0].item() - 1)


@pytest.mark.parametrize('lane', LANES)
def test_historical_meta_lion_uses_old_moment(lane):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(()))])
    optimizer = hf(lane, model, base='SGD', meta='Lion')
    optimizer.step(model, model[0])  # h=-.1, zero initial meta gradient
    optimizer.step(model, model[0])  # hypergrad=-.1, beta increases by .01
    beta = optimizer.beta[0].item()
    moment = optimizer.momentum_meta[0].item()
    assert moment == pytest.approx(-0.01)
    optimizer.step(model, -model[0])  # h=-.2 => hypergrad=.2, old-moment sign stays negative
    assert optimizer.beta[0].item() == pytest.approx(beta + 0.01)
    assert optimizer.momentum_meta[0].item() == pytest.approx(0.011)


@pytest.mark.parametrize('lane', LANES)
def test_none_gradients_stable_groups_and_inactive_state(lane):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(())) for _ in range(3)])
    optimizer = hf(lane, model, base='AdamW', groups='layerwise')
    for active in [2, 0, 1]:
        optimizer.zero_grad()
        before = optimizer.state_dict()['state']
        weights = [p.clone() for p in model]
        model[active].grad = torch.ones_like(model[active])
        optimizer.step(model)
        for i in set(range(3)) - {active}:
            assert torch.equal(model[i], weights[i])
            for field in ['h_condenced', 'trace_base', 'momentum_base']:
                assert torch.equal(getattr(optimizer, field)[i], before[field][i])
            for field in ['beta', 'trace_meta', 'momentum_meta']:
                assert torch.equal(getattr(optimizer, field)[0][i], before[field][0][i])
    optimizer.zero_grad()
    before = optimizer.state_dict()
    optimizer.step(model)
    equal(before, optimizer.state_dict())


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('choice', ['HF', 'AdamW'])
def test_cpu_amp_scaler_unscale_and_step(lane, choice):
    load(lane, 'HF')
    model = torch.nn.Linear(2, 1)
    optimizer = load(lane, 'build_optimizer').build_optimizer(model, args(choice))
    scaler = torch.amp.GradScaler('cpu')
    before = model.weight.clone()
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer, model)
    scaler.update()
    assert not torch.equal(before, model.weight)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('choice', ['HF', 'AdamW', 'SGD', 'SGDm', 'Adam', 'Lion'])
def test_exact_state_scheduler_resume(lane, choice):
    if lane != 'imagenet' and choice not in ('HF', 'AdamW'):
        pytest.skip('not offered')
    load(lane, 'HF')
    builder = load(lane, 'build_optimizer').build_optimizer
    a = torch.nn.Linear(2, 1)
    settings = args(choice, warmup_steps=1, total_steps=6, min_lr=0.001)
    first = builder(a, settings)
    for _ in range(2):
        first.step(a, a(torch.ones(1, 2)).sum())
    b = copy.deepcopy(a)
    second = builder(b, settings)
    second.load_state_dict(first.state_dict())
    for _ in range(3):
        first.step(a, a(torch.ones(1, 2)).sum())
        second.step(b, b(torch.ones(1, 2)).sum())
        equal(first.state_dict(), second.state_dict())
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)
    bad = first.state_dict()
    bad['config']['optimizer'] = 'invalid'
    with pytest.raises(ValueError, match='incompatible'):
        second.load_state_dict(bad)


@pytest.mark.parametrize('lane', LANES)
def test_diagnostics(lane):
    cls = load(lane, 'HF').HF
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match='unsupported base'):
        cls(model, 'scalar', .1, {'alg': 'no'}, {'alg': 'fixed'}, 1)
    with pytest.raises(ValueError, match='missing base'):
        cls(model, 'scalar', .1, {'alg': 'SGD'}, {'alg': 'fixed'}, 1)
    for alpha in [0., -1., float('inf'), float('nan')]:
        with pytest.raises(ValueError, match='alpha0'):
            hf(lane, model, alpha0=alpha)
    with pytest.raises(ValueError, match='unsupported optimizer'):
        load(lane, 'build_optimizer').build_optimizer(model, args('invalid'))
    model.weight.grad = torch.ones_like(model.weight).to_sparse()
    before = model.weight.clone()
    with pytest.raises(ValueError, match='sparse'):
        hf(lane, model).step(model)
    assert torch.equal(before, model.weight)


@pytest.mark.parametrize('lane', LANES)
def test_schedule_bounds_and_native_arguments(lane):
    build = load(lane, 'build_optimizer').build_optimizer
    optimizer = build(torch.nn.Linear(1, 1), args('AdamW', total_steps=4, warmup_steps=1, min_lr=.01))
    assert optimizer.config['scheduler'] == 'cosine'
    assert optimizer.get_learning_rates() == [0.]
    for _ in range(8):
        optimizer.optimizer.step()
        optimizer.scheduler.step()
    assert optimizer.get_learning_rates() == [.01]
    for overrides in [dict(total_steps=1, warmup_steps=1), dict(total_steps=0),
                      dict(total_steps=4, min_lr=.2)]:
        with pytest.raises(ValueError):
            build(torch.nn.Linear(1, 1), args('AdamW', **overrides))
    if lane != 'tinystories':
        with pytest.raises(ValueError, match='require total_steps'):
            build(torch.nn.Linear(1, 1), args('AdamW', warmup_steps=1))


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('choice', ['HF', 'AdamW'])
def test_amp_overflow_skips_entire_optimizer_state(lane, choice):
    load(lane, 'HF')
    model = torch.nn.Linear(1, 1)
    optimizer = load(lane, 'build_optimizer').build_optimizer(
        model, args(choice, warmup_steps=1, total_steps=5))
    scaler = torch.amp.GradScaler('cpu')
    before = optimizer.state_dict()
    weights = copy.deepcopy(model.state_dict())
    scaler.scale(model(torch.ones(1, 1)).sum() * float('inf')).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer, model)
    scaler.update()
    equal(before, optimizer.state_dict())
    equal(weights, model.state_dict())


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('choice', ['HF', 'AdamW'])
def test_invalid_state_container_diagnostic_without_mutation(lane, choice):
    load(lane, 'HF')
    model = torch.nn.Linear(1, 1)
    optimizer = load(lane, 'build_optimizer').build_optimizer(model, args(choice))
    before = optimizer.state_dict()
    invalid = [None, [], {}]
    if choice == 'HF':
        malformed = optimizer.state_dict()
        malformed['state'] = None
        invalid.append(malformed)
    for state in invalid:
        with pytest.raises(ValueError):
            optimizer.load_state_dict(state)
        equal(before, optimizer.state_dict())


@pytest.mark.parametrize('lane', LANES)
def test_historical_negative_actual_step_trace_includes_decay(lane):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((), dtype=torch.double))])
    optimizer = load(lane, 'HF').HF(
        model, 'scalar', .1, dict(alg='SGD', weight_decay=.2), dict(alg='fixed'), .5)
    optimizer.step(model, 2 * model[0])
    assert model[0].item() == pytest.approx(.78)
    assert optimizer.h_condenced[0].item() == pytest.approx(-.22)
    optimizer.step(model, -model[0])
    assert model[0].item() == pytest.approx(.8644)
    assert optimizer.h_condenced[0].item() == pytest.approx(.5 * .98 * -.22 + .0844)


@pytest.mark.parametrize('lane', LANES)
def test_historical_sgdm_steps_with_previous_moment(lane):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((), dtype=torch.double))])
    optimizer = hf(lane, model, base='SGDm', meta='fixed')
    optimizer.step(model, 2 * model[0])
    assert model[0].item() == 1.
    optimizer.step(model, -model[0])
    assert model[0].item() == pytest.approx(.98)
    assert optimizer.momentum_base[0].item() == pytest.approx(.08)
