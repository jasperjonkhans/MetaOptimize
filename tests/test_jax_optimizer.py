"""Real CPU JAX regressions for B03/B08, M01/M02, R01, W01 and T01."""
import importlib
import math
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
pytest.importorskip("chex")

import numpy as np

jax_optimizer = importlib.import_module("meta_optimize.MetaOptimize_AdamW_Lion_hg_jax")


def assert_tree_close(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for a, e in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("left,right,expected", [
    (jnp.array(1.), jnp.array(3.), 3.),
    (jnp.array([1., 2.]), jnp.array([3., 4.]), 11.),
    (jnp.array([[1., 2.], [3., 4.]]), jnp.array([[5., 6.], [7., 8.]]), 70.),
    ({"a": (jnp.array([1., 2.]), jnp.array(2.)), "b": jnp.array([[1., 2.], [3., 4.]])},
     {"a": (jnp.array([3., 4.]), jnp.array(3.)), "b": jnp.array([[5., 6.], [7., 8.]])}, 87.),
])
def test_inner_product_corresponding_leaves_and_zero_trace(compiled, left, right, expected):
    inner = jax.jit(jax_optimizer.inner_product) if compiled else jax_optimizer.inner_product
    result = inner(left, right)
    assert result.shape == ()
    np.testing.assert_allclose(result, expected)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, left)
    np.testing.assert_array_equal(inner(zeros, right), 0.)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("callable_mask", [False, True])
@pytest.mark.parametrize("decay", [0., .2])
def test_two_updates_masked_weights_and_trace_hand_calculation(compiled, callable_mask, decay):
    mask = {"pair": (True, False), "matrix": False}
    opt = jax_optimizer.MetaOptimize_AdamW_Lion_hg(
        alpha0=.1, meta_stepsize=.2, gamma=.8, b1=.5, b2=.5,
        meta_b1=.5, meta_c1=.9, weight_decay=decay,
        weight_decay_mask=(lambda _: mask) if callable_mask else mask,
    )
    params = {"pair": (jnp.array(1.), jnp.array([1., 1.])), "matrix": jnp.ones((2, 2))}
    state = opt.init(params)
    update = jax.jit(opt.update) if compiled else opt.update
    # Step 1: mu=1, nu=2, corrected moments=2,4, z=0, alpha=.1.
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 2.), params)
    updates, state = update(grads, state, params)
    effective_decay = {"pair": (decay, 0.), "matrix": 0.}
    expected_updates = jax.tree_util.tree_map(
        lambda p, d: jnp.full_like(p, -.1 * (1. + d)), params, effective_decay)
    assert_tree_close(updates, expected_updates)
    assert_tree_close(state.mu, jax.tree_util.tree_map(jnp.ones_like, params))
    assert_tree_close(state.nu, jax.tree_util.tree_map(lambda p: jnp.full_like(p, 2.), params))
    assert_tree_close(state.h, jax.tree_util.tree_map(lambda p: jnp.full_like(p, .1), params))
    np.testing.assert_allclose(state.beta, math.log(.1), rtol=1e-6)
    params = optax.apply_updates(params, updates)
    # Step 2: mu=0, nu=1.5, corrected moments=0,2, z=-.7.
    # Seven scalar components each have old h=.1 and grad=-1.
    grads = jax.tree_util.tree_map(lambda p: -jnp.ones_like(p), params)
    updates, state = update(grads, state, params)
    alpha = .1 * math.exp(-.2)
    expected_updates = jax.tree_util.tree_map(lambda p, d: -alpha * d * p, params, effective_decay)
    expected_h = jax.tree_util.tree_map(
        lambda p, d: jnp.full_like(p, .8 * (1 - alpha * d) * .1 - alpha / math.sqrt(2)),
        params, effective_decay)
    assert_tree_close(updates, expected_updates)
    assert_tree_close(state.h, expected_h)
    assert_tree_close(state.mu, jax.tree_util.tree_map(jnp.zeros_like, params))
    assert_tree_close(state.nu, jax.tree_util.tree_map(lambda p: jnp.full_like(p, 1.5), params))
    np.testing.assert_allclose(state.meta_momentum, -.35, rtol=1e-6)
    np.testing.assert_allclose(state.beta, math.log(.1) - .2, rtol=1e-6)
    assert int(state.count) == 2


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("b1", [0., .5, .9])
def test_normalized_adam_moments_with_meta_disabled(compiled, b1):
    opt = jax_optimizer.MetaOptimize_AdamW_Lion_hg(alpha0=.1, meta_stepsize=0., b1=b1, b2=.5)
    params = jnp.array(1.)
    state = opt.init(params)
    update = jax.jit(opt.update) if compiled else opt.update
    mu = nu = 0.
    for count, grad in enumerate([2., -1., .5], start=1):
        updates, state = update(jnp.array(grad), state, params)
        mu = b1 * mu + (1 - b1) * grad
        nu = .5 * nu + .5 * grad ** 2
        expected = -.1 * (mu / (1 - b1 ** count)) / (math.sqrt(nu / (1 - .5 ** count)) + 1e-8)
        np.testing.assert_allclose(updates, expected, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(state.mu, mu, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(state.nu, nu, rtol=2e-5)
        params = optax.apply_updates(params, updates)


@pytest.mark.parametrize("compiled", [False, True])
def test_lion_uses_new_momentum_and_new_learning_rate(compiled):
    opt = jax_optimizer.MetaOptimize_AdamW_Lion_hg(
        alpha0=.1, meta_stepsize=.1, meta_b1=.99, meta_c1=.9)
    params = jnp.array(1.)
    state = opt.init(params)._replace(h=jnp.array(1.), meta_momentum=jnp.array(1.))
    update = jax.jit(opt.update) if compiled else opt.update
    updates, state = update(jnp.array(-8.5), state, params)
    # New M=.905 gives sign(.9*.905-.85)<0; old M would give >0.
    np.testing.assert_allclose(state.meta_momentum, .905, rtol=1e-6)
    np.testing.assert_allclose(state.beta, math.log(.1) - .1, rtol=1e-6)
    np.testing.assert_allclose(updates, .1 * math.exp(-.1), rtol=2e-5)


@pytest.mark.parametrize("mask", [None, {"x": True}, lambda _: {"x": True}])
@pytest.mark.parametrize("compiled", [False, True])
def test_nonzero_decay_requires_params(mask, compiled):
    opt = jax_optimizer.MetaOptimize_AdamW_Lion_hg(weight_decay=.1, weight_decay_mask=mask)
    grads = {"x": jnp.array(1.)}
    update = jax.jit(opt.update) if compiled else opt.update
    with pytest.raises(ValueError, match="params are required"):
        update(grads, opt.init(grads))
    with pytest.raises(ValueError, match="params are required"):
        jax_optimizer.add_decayed_weights(grads, None, .1, mask)


@pytest.mark.parametrize("compiled", [False, True])
def test_zero_decay_can_omit_params(compiled):
    opt = jax_optimizer.MetaOptimize_AdamW_Lion_hg()
    grads = {"x": jnp.array([1., 2.])}
    state = opt.init(grads)
    update = jax.jit(opt.update) if compiled else opt.update
    for _ in range(2):
        updates, state = update(grads, state)
        assert np.all(np.isfinite(updates["x"]))
    assert_tree_close(jax_optimizer.add_decayed_weights(grads, None, 0.), grads)


def test_demo_jit_closes_over_optimizer_and_preserves_state_tree():
    opt = jax_optimizer.MetaOptimize_AdamW_Lion_hg()
    params = {"w": jnp.array([1., -1.]), "b": jnp.array(0.)}
    state = jax_optimizer.TrainState(params, opt.init(params))
    structure = jax.tree_util.tree_structure(state)
    def loss(p, x, y):
        return jnp.mean((x @ p["w"] + p["b"] - y) ** 2)
    step = jax_optimizer.make_train_step(opt, loss)
    for _ in range(3):
        state = step(state, jnp.ones((2, 2)), jnp.ones(2))
        assert jax.tree_util.tree_structure(state) == structure
    assert int(state.opt_state.count) == 3


def test_demo_shuffle_uses_supplied_seed():
    x = np.arange(24)
    def order(seed):
        return np.concatenate([batch for batch, _ in jax_optimizer.get_batches(
            x, x, 5, np.random.default_rng(seed))])
    np.testing.assert_array_equal(order(12), order(12))
    assert not np.array_equal(order(12), order(13))
    np.testing.assert_array_equal(np.sort(order(12)), x)


def test_core_import_does_not_require_demo_dependencies():
    code = '''
import importlib.abc
import sys
class NoDemoImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'haiku', 'tensorflow', 'tensorflow_datasets'}:
            raise AssertionError('optional demo import: ' + fullname)
sys.meta_path.insert(0, NoDemoImports())
from meta_optimize.MetaOptimize_AdamW_Lion_hg_jax import MetaOptimize_AdamW_Lion_hg
import jax.numpy as jnp
opt = MetaOptimize_AdamW_Lion_hg()
opt.update(jnp.array(1.), opt.init(jnp.array(1.)))
'''
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ), timeout=90)
