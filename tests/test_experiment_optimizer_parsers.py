"""Real vision-parser integration plus historical native Namespace compatibility.

EXPERIMENT_VISION_PARSER_ROOT may point at the vision lane's codes/MetaOptimize
while its parser changes await integration. Builders always come from this tree.
"""
import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from test_experiment_optimizers import ROOT, load


def load_vision_parser(lane, monkeypatch):
    if lane == 'imagenet':
        pytest.importorskip('torchvision', exc_type=ModuleNotFoundError,
                            reason='Loading the actual ImageNet parser requires optional torchvision')
    root = Path(os.environ.get('EXPERIMENT_VISION_PARSER_ROOT', ROOT))
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(f'optimizer_parser_{lane}', root / lane / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parse = module.parse_args if lane == 'cifar10' else module.parser.parse_args
    return lane, parse


@pytest.fixture(params=['cifar10', 'imagenet'])
def vision_parser(request, monkeypatch):
    return load_vision_parser(request.param, monkeypatch)


def test_actual_new_parser_offered_optimizer_matrix(vision_parser):
    lane, parse = vision_parser
    choices = ['HF', 'AdamW'] if lane == 'cifar10' else ['HF', 'AdamW', 'SGD', 'SGDm', 'Adam', 'Lion']
    load(lane, 'HF')
    build = load(lane, 'build_optimizer').build_optimizer
    for choice in choices:
        settings = parse(['--optimizer', choice])
        assert hasattr(settings, 'base_lr') and hasattr(settings, 'base_momentum')
        model = torch.nn.Linear(2, 1)
        optimizer = build(model, settings)
        json.dumps(optimizer.config)
        before = model.weight.detach().clone()
        for _ in range(2):
            optimizer.step(model, model(torch.ones(1, 2)).sum())
        assert not torch.equal(before, model.weight)
        if choice in ['SGD', 'SGDm', 'Adam']:
            assert optimizer.config['lr'] == settings.base_lr
            assert optimizer.config['weight_decay'] == settings.weight_decay_base
            assert optimizer.config['scheduler'] == ('constant' if choice == 'Adam' else 'StepLR')
        elif choice != 'HF':
            assert optimizer.config['scheduler'] == 'constant'


def test_actual_new_parser_explicit_values_and_schedules(vision_parser):
    lane, parse = vision_parser
    choices = ['AdamW'] if lane == 'cifar10' else ['SGD', 'SGDm', 'Adam', 'AdamW', 'Lion']
    build = load(lane, 'build_optimizer').build_optimizer
    for choice in choices:
        settings = parse(['--optimizer', choice, '--base-lr', '.037', '--momentum', '.42',
                          '--alpha0', '.017', '--weight-decay-base', '.023',
                          '--momentum-param-base', '.81', '--normalizer-param-base', '.97',
                          '--warmup-steps', '1', '--total-steps', '5'])
        model = torch.nn.Linear(2, 1)
        optimizer = build(model, settings)
        assert optimizer.config['weight_decay'] == .023
        if choice in ['SGD', 'SGDm', 'Adam']:
            assert optimizer.config['lr'] == .037
            assert optimizer.get_learning_rates() == [.037]
            assert optimizer.config['scheduler'] == ('constant' if choice == 'Adam' else 'StepLR')
            if choice == 'SGDm':
                assert optimizer.config['momentum'] == .42
                assert optimizer.param_groups[0]['momentum'] == .42
        else:
            assert optimizer.config['lr'] == .017
            assert optimizer.config['scheduler'] == 'cosine'
            assert optimizer.config['warmup_steps'] == 1
            assert optimizer.config['total_steps'] == 5
            assert optimizer.get_learning_rates() == [0.]
        if choice in ['Adam', 'AdamW']:
            assert optimizer.config['betas'] == [.81, .97]
        optimizer.step(model, model(torch.ones(1, 2)).sum())
        if optimizer.scheduler is not None and choice in ['AdamW', 'Lion']:
            assert optimizer.get_learning_rates() == [.017]
        if choice in ['SGD', 'SGDm']:
            # Exercise the actual configured StepLR boundary without a long loop.
            optimizer.scheduler.last_epoch = optimizer.config['step_size'] - 1
            optimizer.step(model, model(torch.ones(1, 2)).sum())
            assert optimizer.get_learning_rates() == pytest.approx([.0037])


@pytest.mark.parametrize('choice', ['SGD', 'SGDm', 'Adam'])
@pytest.mark.parametrize('canonical', [False, True])
def test_imagenet_legacy_namespace_and_canonical_precedence(choice, canonical):
    settings = Namespace(optimizer=choice, lr=.043, momentum=.62, weight_decay=.031)
    if canonical:
        settings.base_lr = .057
        settings.base_momentum = .48
    model = torch.nn.Linear(2, 1)
    optimizer = load('imagenet', 'build_optimizer').build_optimizer(model, settings)
    expected_lr, expected_momentum = (.057, .48) if canonical else (.043, .62)
    assert optimizer.config['lr'] == expected_lr
    assert optimizer.get_learning_rates() == [expected_lr]
    assert optimizer.config['weight_decay'] == .031
    if choice == 'SGDm':
        assert optimizer.config['momentum'] == expected_momentum
    for _ in range(2):
        optimizer.step(model, model(torch.ones(1, 2)).sum())


@pytest.mark.parametrize('flag', ['--wd', '--weight-decay', '--weight-decay-base'])
def test_actual_imagenet_parser_decay_aliases(monkeypatch, flag):
    lane, parse = load_vision_parser('imagenet', monkeypatch)
    for choice in ['SGD', 'SGDm', 'Adam']:
        settings = parse(['--optimizer', choice, flag, '.061'])
        optimizer = load(lane, 'build_optimizer').build_optimizer(torch.nn.Linear(1, 1), settings)
        assert optimizer.config['weight_decay'] == .061
        assert optimizer.param_groups[0]['weight_decay'] == .061
