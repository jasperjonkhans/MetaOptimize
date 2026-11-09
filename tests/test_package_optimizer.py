"""Hermetic CPU regressions for the manual package optimizer (real PyTorch)."""
import copy
import io
import math
import subprocess
import sys
import unittest
from unittest import mock

import torch

from meta_optimize import MetaOptimize_AdamW_Lion_hg as Optimizer


def parameters(count=3, dtype=torch.float64):
    return [torch.nn.Parameter(torch.tensor([1., -2.], dtype=dtype)) for _ in range(count)]


class PackageOptimizerTests(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.device, right.device)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_layerwise_missing_first_gradient_original_repro(self):
        ps = parameters(2)
        opt = Optimizer(ps, stepsize_blocks='layerwise', meta_stepsize=0)
        for block, lr in zip(opt.meta_state, [.1, .01]):
            block['beta'].fill_(math.log(lr))
        ps[1].grad = torch.ones_like(ps[1])
        opt.step()
        torch.testing.assert_close(ps[0], torch.tensor([1., -2.], dtype=torch.float64))
        torch.testing.assert_close(ps[1], torch.tensor([.99, -2.01], dtype=torch.float64))
        self.assertEqual(opt.state[ps[0]]['step'], 0)
        self.assertEqual(opt.state[ps[1]]['step'], 1)

    def test_fixed_membership_missing_first_middle_last(self):
        # Three fixed blocks each lose one different member, never shifting rates.
        ps = parameters(9)
        opt = Optimizer(ps, stepsize_blocks=[3, 3, 3], meta_stepsize=.1)
        rates = [.1, .01, .001]
        before = []
        for block, lr in zip(opt.meta_state, rates):
            block['beta'].fill_(math.log(lr))
            before.append(block['beta'].item())
        for i, p in enumerate(ps):
            opt.state[p]['h'].fill_(1)
            if i not in (0, 4, 8):
                p.grad = torch.ones_like(p)
        opt.step()
        for i, p in enumerate(ps):
            lr = math.exp(before[i // 3]) if i not in (0, 4, 8) else 0
            torch.testing.assert_close(p, torch.tensor([1-lr/(1+opt.eps), -2-lr/(1+opt.eps)], dtype=p.dtype))
        for old, block in zip(before, opt.meta_state):
            self.assertAlmostEqual(block['beta'].item(), old+.1, places=6)
            self.assertGreater(block['meta_momentum'].item(), 0)

    def test_scalar_partial_block_finishes_original_repro(self):
        ps = parameters(2)
        opt = Optimizer(ps, meta_stepsize=.1)
        opt.state[ps[0]]['h'].fill_(1)
        ps[0].grad = torch.ones_like(ps[0])
        beta = opt.meta_state[0]['beta'].item()
        opt.step()
        self.assertAlmostEqual(opt.meta_state[0]['beta'].item(), beta+.1, places=6)

    def test_fully_inactive_blocks_freeze_including_meta_momentum(self):
        ps = parameters(3)
        opt = Optimizer(ps, stepsize_blocks='layerwise', weight_decay=.1)
        for block in opt.meta_state:
            block['meta_momentum'].fill_(2)
        before = opt.state_dict()
        ps[1].grad = torch.ones_like(ps[1])
        opt.step()
        after = opt.state_dict()
        for i in (0, 2):
            self.assert_tree_equal(before['state'][i], after['state'][i])
            self.assert_tree_equal(before['meta_state'][i], after['meta_state'][i])
            torch.testing.assert_close(ps[i], torch.tensor([1., -2.], dtype=ps[i].dtype))

    def test_zero_gradient_is_not_missing_gradient(self):
        ps = parameters(2)
        opt = Optimizer(ps, stepsize_blocks='layerwise', weight_decay=.5)
        for p in ps:
            opt.state[p]['h'].fill_(1)
            opt.state[p]['base_momentum'].fill_(1)
        ps[1].grad = torch.zeros_like(ps[1])
        before = opt.state_dict()
        opt.step()
        self.assert_tree_equal(before['state'][0], opt.state_dict()['state'][0])
        self.assertEqual(opt.state[ps[1]]['step'], 1)
        self.assertFalse(torch.equal(ps[0], ps[1]))
        self.assertFalse(torch.equal(opt.state[ps[0]]['h'], opt.state[ps[1]]['h']))

    def test_invalid_constructor_before_state_allocation(self):
        invalid = [dict(stepsize_blocks=b) for b in ([1], [3], [0, 2], [-1, 3], [.5, 1.5], [], [True, 1], 'unknown', None)]
        invalid += [dict(alpha0=x) for x in (0, -.1, float('nan'), float('inf'))]
        invalid += [{name: x} for name in ('b1', 'b2', 'meta_b1', 'meta_c1') for x in (-.1, 1, float('nan'), float('inf'))]
        invalid += [{name: x} for name in ('gamma', 'meta_stepsize', 'weight_decay') for x in (-.1, float('nan'), float('inf'))]
        invalid += [dict(gamma=1.1)]
        ps = parameters(2)
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), mock.patch('torch.zeros_like', side_effect=AssertionError('allocated before validation')):
                with self.assertRaises(ValueError):
                    Optimizer(ps, **kwargs)
        with self.assertRaises(ValueError):
            Optimizer([])
        for p in ps:
            torch.testing.assert_close(p, torch.tensor([1., -2.], dtype=p.dtype))

    def test_valid_boundaries(self):
        for blocks in ('scalar', 'layerwise', [1, 1]):
            opt = Optimizer(parameters(2), stepsize_blocks=blocks, meta_stepsize=0, gamma=1,
                            b1=0, b2=0, meta_b1=0, meta_c1=0, weight_decay=0)
            opt.step()
        for alpha in (1e-300, 1e300):
            opt = Optimizer(parameters(1), alpha0=alpha)
            self.assertTrue(torch.isfinite(opt.meta_state[0]['beta']).item())

    def test_duplicate_and_invalid_parameters(self):
        p = parameters(1)[0]
        for ps in ([p, p], [torch.tensor([1])], [object()]):
            with self.subTest(ps=ps), self.assertRaises(ValueError):
                Optimizer(ps)

    def test_sparse_rejected_before_any_mutation(self):
        ps = parameters(2)
        opt = Optimizer(ps, weight_decay=.1)
        ps[0].grad = torch.ones_like(ps[0])
        ps[1].grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.], dtype=ps[1].dtype), size=ps[1].shape)
        before = opt.state_dict()
        with self.assertRaisesRegex(ValueError, 'sparse'):
            opt.step()
        self.assert_tree_equal(before, opt.state_dict())
        for p in ps:
            torch.testing.assert_close(p, torch.tensor([1., -2.], dtype=p.dtype))

    def test_fixed_lr_matches_real_adamw_changing_float64_gradients(self):
        for b1 in (0., .5, .9):
            for decay in (0., .1):
                with self.subTest(b1=b1, decay=decay):
                    p = parameters(1)[0]
                    q = torch.nn.Parameter(p.detach().clone())
                    opt = Optimizer([p], alpha0=.03, b1=b1, b2=.8, weight_decay=decay, meta_stepsize=0)
                    # Use the exact float32-encoded package learning rate.
                    lr = opt.meta_state[0]['beta'].exp().item()
                    reference = torch.optim.AdamW([q], lr=lr, betas=(b1, .8), eps=opt.eps, weight_decay=decay)
                    for values in ([1., -2.], [-3., .5], [0., 4.], [.7, -.2], [2., 3.]):
                        p.grad = torch.tensor(values, dtype=p.dtype)
                        q.grad = p.grad.clone()
                        opt.step()
                        reference.step()
                        # Decay and trace scalar arithmetic historically uses float32.
                        torch.testing.assert_close(p, q, rtol=1e-9, atol=1e-9)
                        state = reference.state[q]
                        t = opt.state[p]['step']
                        torch.testing.assert_close(opt.state[p]['base_momentum'], state['exp_avg']/(1-b1**t), rtol=1e-13, atol=1e-14)
                        torch.testing.assert_close(opt.state[p]['base_normalizer'], state['exp_avg_sq']/(1-.8**t), rtol=1e-13, atol=1e-14)

    def test_new_momentum_lion_order_opposes_old_momentum_rule(self):
        p = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        opt = Optimizer([p], alpha0=.1, meta_stepsize=.1, meta_b1=.99, meta_c1=.9)
        opt.state[p]['h'].fill_(1)
        opt.meta_state[0]['meta_momentum'].fill_(1)
        p.grad = torch.tensor(-8.5, dtype=p.dtype)
        beta = opt.meta_state[0]['beta'].item()
        opt.step()
        # The first multiplication retains the historical float32 scalar rounding.
        expected_momentum = float(torch.tensor(1., dtype=torch.float32)*.99) - .085
        self.assertAlmostEqual(opt.meta_state[0]['meta_momentum'].item(), expected_momentum, places=12)
        self.assertLess(.9*.905 + .1*(-8.5), 0)
        self.assertGreater(.9*1 + .1*(-8.5), 0)
        self.assertAlmostEqual(opt.meta_state[0]['beta'].item(), beta-.1, places=6)
        # Weight update used old alpha, not exp(beta_new).
        self.assertAlmostEqual(p.item(), 1+float(torch.tensor(beta).exp())*8.5/(8.5+opt.eps), places=12)

    def test_hand_computed_trace_and_moment_sequence(self):
        for decay in (0., .2):
            with self.subTest(decay=decay):
                p = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
                opt = Optimizer([p], alpha0=.1, meta_stepsize=.07, gamma=.8,
                                b1=.5, b2=.25, meta_b1=.6, meta_c1=.7, weight_decay=decay)
                w, h, m, v, meta_m = 1., 0., 0., 0., 0.
                beta = opt.meta_state[0]['beta'].item()
                for step, grad in enumerate((2., -1., .5, -3.), 1):
                    alpha = float(torch.tensor(beta, dtype=torch.float32).exp())
                    m = .5*m + .5*grad
                    v = .25*v + .75*grad*grad
                    m_hat, v_hat = m/(1-.5**step), v/(1-.25**step)
                    z = h*grad
                    w = w*(1-alpha*decay)-alpha*m_hat/(math.sqrt(v_hat)+opt.eps)
                    h = .8*(1-alpha*decay)*h + alpha*grad/(math.sqrt(v_hat)+opt.eps)
                    meta_m = .6*meta_m + .4*z
                    u = .7*meta_m + .3*z
                    beta = float(torch.tensor(beta + .07*((u > 0)-(u < 0)), dtype=torch.float32))
                    p.grad = torch.tensor(grad, dtype=p.dtype)
                    opt.step()
                    self.assertAlmostEqual(p.item(), w, places=8)
                    self.assertAlmostEqual(opt.state[p]['h'].item(), h, places=7)
                    self.assertAlmostEqual(opt.state[p]['base_momentum'].item(), m_hat, places=12)
                    self.assertAlmostEqual(opt.state[p]['base_normalizer'].item(), v_hat, places=12)
                    self.assertAlmostEqual(opt.meta_state[0]['meta_momentum'].item(), meta_m, places=7)
                    self.assertAlmostEqual(opt.meta_state[0]['beta'].item(), beta, places=6)

    def advance(self, opt, start, end):
        for t in range(start, end):
            for i, p in enumerate(opt.params):
                p.grad = None if (t+i) % 4 == 0 else torch.full_like(p, (t+1)*(-1 if i % 2 else 1)*.17)
            opt.step()

    def test_resume_all_states_and_hyperparameters_no_aliasing(self):
        ps = parameters(3)
        opt = Optimizer(ps, stepsize_blocks=[2, 1], alpha0=.02, meta_stepsize=.03,
                        gamma=.8, b1=.6, b2=.7, meta_b1=.4, meta_c1=.5, weight_decay=.1)
        self.advance(opt, 0, 4)
        snapshot = opt.state_dict()
        frozen = copy.deepcopy(snapshot)
        model_weights = [p.detach().clone() for p in ps]
        # Snapshot is serializable with real torch.save/load, without pickle objects.
        buffer = io.BytesIO()
        torch.save(snapshot, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        resumed_ps = [torch.nn.Parameter(p.clone()) for p in model_weights]
        resumed = Optimizer(resumed_ps, stepsize_blocks=[2, 1], alpha0=.9)
        resumed.load_state_dict(saved)
        self.assert_tree_equal(frozen, resumed.state_dict())
        self.advance(opt, 4, 9)
        self.advance(resumed, 4, 9)
        self.assert_tree_equal(opt.state_dict(), resumed.state_dict())
        for p, q in zip(ps, resumed_ps):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assert_tree_equal(frozen, snapshot)  # state_dict must not alias live state
        self.assert_tree_equal(frozen, saved)  # load/steps must not mutate the input
        saved['state'][0]['h'].fill_(999)
        saved['meta_state'][0]['beta'].fill_(999)
        self.assert_tree_equal(opt.state_dict(), resumed.state_dict())

    def test_load_converts_base_dtype_and_preserves_beta_precision(self):
        source = Optimizer(parameters(2), stepsize_blocks='layerwise')
        self.advance(source, 0, 3)
        snapshot = source.state_dict()
        target = Optimizer(parameters(2, dtype=torch.float32), stepsize_blocks='layerwise')
        target.load_state_dict(snapshot)
        for p in target.params:
            self.assertEqual(target.state[p]['h'].dtype, torch.float32)
            self.assertEqual(target.state[p]['h'].device, p.device)
        for block in target.meta_state:
            self.assertEqual(block['beta'].dtype, torch.float32)
            self.assertEqual(block['meta_momentum'].dtype, torch.float64)
        self.assert_tree_equal(snapshot, source.state_dict())

    def test_load_unrepresentable_base_state_is_atomic(self):
        source = Optimizer(parameters(1))
        snapshot = source.state_dict()
        snapshot['state'][0]['h'].fill_(1e100)
        target = Optimizer(parameters(1, dtype=torch.float32), alpha0=.2)
        before = target.state_dict()
        frozen = copy.deepcopy(snapshot)
        with self.assertRaisesRegex(ValueError, 'destination dtype'):
            target.load_state_dict(snapshot)
        self.assert_tree_equal(before, target.state_dict())
        self.assert_tree_equal(frozen, snapshot)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; CPU/dtype conversion covered')
    def test_load_moves_state_to_cuda(self):
        source = Optimizer(parameters(2), stepsize_blocks='layerwise')
        self.advance(source, 0, 3)
        ps = [torch.nn.Parameter(p.detach().cuda()) for p in source.params]
        target = Optimizer(ps, stepsize_blocks='layerwise')
        target.load_state_dict(source.state_dict())
        for p in ps:
            self.assertTrue(target.state[p]['h'].is_cuda)
        for block in target.meta_state:
            self.assertTrue(block['beta'].is_cuda)
            self.assertTrue(block['meta_momentum'].is_cuda)

    def test_invalid_checkpoint_is_atomic_and_does_not_mutate_input(self):
        opt = Optimizer(parameters(3), stepsize_blocks=[2, 1])
        self.advance(opt, 0, 3)
        valid = opt.state_dict()
        corruptions = [
            lambda s: s.update(version=2),
            lambda s: s.update(variant='other'),
            lambda s: s.update(block_sizes=[1, 2]),
            lambda s: s.update(block_sizes=[0, 3]),
            lambda s: s.update(param_shapes=[(3,), (2,), (2,)]),
            lambda s: s.update(param_shapes=[(2,), (2,)]),
            lambda s: s.update(block_sizes=[3]),
            lambda s: s['hyperparameters'].update(b1=1),
            lambda s: s['hyperparameters'].update(alpha0=0),
            lambda s: s['hyperparameters'].update(eps=float('inf')),
            lambda s: s['hyperparameters'].pop('gamma'),
            lambda s: s['state'].pop(),
            lambda s: s['state'][0].update(step=-1),
            lambda s: s['state'][0].update(step=True),
            lambda s: s['state'][0].update(h=torch.zeros(3)),
            lambda s: s['state'][0].update(base_normalizer=-torch.ones(2)),
            lambda s: s['meta_state'].pop(),
            lambda s: s['meta_state'][1].update(num_layers_in_this_block=2),
            lambda s: s['meta_state'][1].update(beta=torch.zeros(1)),
            lambda s: s['meta_state'][1].update(meta_momentum='bad'),
        ]
        for corrupt in corruptions:
            broken = copy.deepcopy(valid)
            corrupt(broken)
            before = copy.deepcopy(broken)
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                opt.load_state_dict(broken)
            self.assert_tree_equal(valid, opt.state_dict())
            self.assert_tree_equal(before, broken)
        for broken in (None, {}, {'version': 1}):
            with self.assertRaises(ValueError):
                opt.load_state_dict(broken)

    def test_loaded_tensors_must_be_finite(self):
        opt = Optimizer(parameters(1))
        for field in ('h', 'base_normalizer', 'base_momentum'):
            snapshot = opt.state_dict()
            snapshot['state'][0][field].fill_(float('nan'))
            with self.assertRaises(ValueError):
                opt.load_state_dict(snapshot)
        for field in ('beta', 'meta_momentum'):
            snapshot = opt.state_dict()
            snapshot['meta_state'][0][field].fill_(float('inf'))
            with self.assertRaises(ValueError):
                opt.load_state_dict(snapshot)

    def test_manual_backward_zero_grad_and_no_implicit_clearing(self):
        p, missing = parameters(2)
        opt = Optimizer([p, missing])
        (p.square().sum()).backward()
        gradient = p.grad.clone()
        opt.step()
        torch.testing.assert_close(p.grad, gradient)
        opt.zero_grad()
        torch.testing.assert_close(p.grad, torch.zeros_like(p))
        self.assertIsNone(missing.grad)
        self.assertNotIsInstance(opt, torch.optim.Optimizer)

    def test_import_needs_no_demo_dependencies_or_downloads(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torchvision', 'jax', 'optax', 'chex', 'tensorflow', 'haiku'}:
        raise AssertionError('optional demo import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from meta_optimize import MetaOptimize_AdamW_Lion_hg
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
