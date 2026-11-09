"""JAX historical variant of MetaOptimize AdamW/Lion.

Base Adam uses normalized EMA moments and bias correction (epsilon outside the
square root). Meta-Lion updates its momentum before choosing the sign, then
uses the *new* exp(beta) for both the current weight step and trace. The positive
trace injects alpha * grad / (sqrt(v_hat) + eps), not the weight update. Its
retention is gamma * (1 - alpha * effective_leaf_decay). These historical
conventions intentionally differ from the Torch/experimental variants.

Core imports require JAX, Optax and Chex; the guarded MNIST demo additionally
requires NumPy, dm-haiku and TensorFlow and downloads data only when run.
"""

from typing import Any, Callable, NamedTuple, Optional, Union
import chex
import jax
import jax.numpy as jnp
import optax


def MetaOptimize_AdamW_Lion_hg(
    meta_stepsize: float = 1e-3,
    alpha0: float = 1e-3,   # conservative initialization for Adam
    gamma = 0.9999, 
    b1: float = 0.9,        # base-adamw parameter
    b2: float = 0.999,      # base-adamw parameters
    meta_b1: float = 0.99,  # meta-lion parameters
    meta_c1: float = 0.9,   # meta-lion parameters
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:

    def init_fn(params):
        mu = jax.tree_util.tree_map(jnp.zeros_like, params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        h = jax.tree_util.tree_map(jnp.zeros_like, params)
        beta = jnp.array(jnp.log(alpha0), dtype=jnp.float32)
        meta_momentum = jnp.zeros([], jnp.float32)
        return ScaleByState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, h=h, beta=beta, meta_momentum=meta_momentum)
    
    def update_fn(grads, state, params=None):
        eps = 1e-8
        count = state.count + 1

        # base update AdamW
        mu = _update_moment(grads, state.mu, b1, 1)
        nu = _update_moment(grads, state.nu, b2, 2)

        mu_hat = _bias_correction(mu, b1, count)
        nu_hat = _bias_correction(nu, b2, count)

        updates = normalize(mu_hat, nu_hat, eps)
        decay = _effective_decay(grads, params, weight_decay, weight_decay_mask)
        updates = _add_decay(updates, params, decay)
    
        meta_grad = inner_product(state.h, grads)
        
        #meta Lion:
        meta_momentum = _update_moment_scalar(meta_grad, state.meta_momentum, meta_b1, 1)
        meta_update = jnp.sign(meta_c1*meta_momentum + (1-meta_c1)*meta_grad)
        
        beta = state.beta + meta_stepsize * meta_update
        alpha = jnp.exp(beta)
        updates = jax.tree_util.tree_map(lambda u: -alpha * u, updates)  # negate because Optax expects "descent"
        
        h = jax.tree_util.tree_map(
            lambda h_, g, v, d: gamma * (1.0 - alpha * d) * h_
            + alpha * g / (jnp.sqrt(v) + eps),
            state.h, grads, nu_hat, decay,
        )
    
        new_state = ScaleByState(count=count, mu=mu, nu=nu, h=h, beta=beta, meta_momentum=meta_momentum)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByState(NamedTuple):
  count: chex.Array  
  mu: optax.Updates
  nu: optax.Updates
  h: optax.Updates
  beta: chex.Array
  meta_momentum: chex.Array


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)

def _update_moment_scalar(updates, moments, decay, order):
  return decay*moments +(1-decay)*(updates**order)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_util.tree_map(lambda t: t / beta.astype(t.dtype), moment)

def _effective_decay(updates, params, weight_decay, weight_decay_mask):
  if weight_decay == 0:
    return jax.tree_util.tree_map(lambda _: 0.0, updates)
  if params is None:
    raise ValueError("params are required when weight_decay is nonzero")
  mask = weight_decay_mask(params) if callable(weight_decay_mask) else weight_decay_mask
  if mask is None:
    return jax.tree_util.tree_map(lambda _: weight_decay, updates)
  return jax.tree_util.tree_map(
      lambda _, m: jnp.where(m, weight_decay, 0.0), updates, mask)


def _add_decay(updates, params, decay):
  if params is None:
    return updates
  return jax.tree_util.tree_map(lambda g, p, d: g + d * p, updates, params, decay)


def add_decayed_weights(updates, params, weight_decay, weight_decay_mask=None):
  decay = _effective_decay(updates, params, weight_decay, weight_decay_mask)
  return _add_decay(updates, params, decay)

def normalize(numerator, denum_square, eps):
  def normalize__(m, v):
      return m / (jnp.sqrt(v) + eps)
  return jax.tree_util.tree_map(normalize__, numerator, denum_square)

def inner_product(tree1, tree2):
  products = jax.tree_util.tree_map(jnp.vdot, tree1, tree2)
  return sum(jax.tree_util.tree_leaves(products), jnp.array(0.0))


# MNIST demo helpers; importing the optimizer never imports demo dependencies.
class TrainState(NamedTuple):
    params: Any
    opt_state: optax.OptState


def make_train_step(optimizer, loss_fn):
    """Close over functions so only arrays/state cross the JIT boundary."""
    @jax.jit
    def train_step(state: TrainState, x, y):
        grads = jax.grad(loss_fn)(state.params, x, y)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return TrainState(new_params, new_opt_state)
    return train_step


def get_batches(x, y, batch_size, rng):
    indices = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


def main(seed=0):
    import haiku as hk
    import numpy as np
    from tensorflow.keras.datasets import mnist

    def mlp_fn(x):
        return hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.Linear(256), jax.nn.relu,
            hk.Linear(10),
        ])(x)

    net = hk.transform(mlp_fn)

    def compute_loss(params, x, y):
        logits = net.apply(params, None, x)
        labels = jax.nn.one_hot(y, 10)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))

    def compute_accuracy(params, x, y):
        logits = net.apply(params, None, x)
        return jnp.mean(jnp.argmax(logits, axis=-1) == y)

    rng = jax.random.PRNGKey(seed)
    shuffle_rng = np.random.default_rng(seed)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = jnp.array(x_train.reshape(-1, 784).astype(np.float32) / 255.)
    x_test = jnp.array(x_test.reshape(-1, 784).astype(np.float32) / 255.)
    y_train, y_test = jnp.array(y_train), jnp.array(y_test)

    params = net.init(rng, jnp.zeros((1, 784), jnp.float32))
    optimizer = MetaOptimize_AdamW_Lion_hg(
        meta_stepsize=0.001, alpha0=1e-3, gamma=0.9999, b1=0.9, b2=0.999,
        meta_b1=0.99, meta_c1=0.9, weight_decay=0.1,
    )
    state = TrainState(params, optimizer.init(params))
    train_step = make_train_step(optimizer, compute_loss)

    for epoch in range(1, 11):
        for xb, yb in get_batches(x_train, y_train, 128, shuffle_rng):
            state = train_step(state, xb, yb)
        train_acc = compute_accuracy(state.params, x_train, y_train)
        test_acc = compute_accuracy(state.params, x_test, y_test)
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MetaOptimize JAX MNIST demo")
    parser.add_argument("--seed", type=int, default=0)
    main(seed=parser.parse_args().seed)
