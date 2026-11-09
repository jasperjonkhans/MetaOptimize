# Optimizer variants — specification v1

This specification names preserved implementation conventions (M01/M02), not
paper-equivalent algorithms. It does not assert equality between the three
families, nor reproduce published results. Repairs to validation, missing
gradients, synchronization and masks do not authorize changing these equations.
Source definitions are `meta_optimize/optimizer.py`,
`meta_optimize/MetaOptimize_AdamW_Lion_hg_jax.py`, and the three experiment
`HF.py` files identified in [provenance](provenance.md).

## Notation

For one dense active parameter tensor: old weight `w`, current gradient `g`,
old trace `h`, log step size `beta`, `alpha=exp(beta)`, decay `k`, meta step
size `eta`, trace discount `gamma`. Moments start at zero. Meta signal `z`
is the sum of `h*g` over corresponding elements in a fixed learning-rate
block, **before** replacing `h`. Each assignment below produces the next
state. Epsilon constants are literal, not interchangeable.

## P: package corrected-moment / new-meta-moment / old-alpha

The public `MetaOptimize_AdamW_Lion_hg` is a manual dense PyTorch optimizer
(checkpoint variant `package_adamw_lion_hg_v1`). In the equations below,
`r=meta_b1` and `c=meta_c1`. For each active parameter's update count `t`:

```
a1 = (1-b1)/(1-b1**t); a2 = (1-b2)/(1-b2**t)
m = (1-a1)*m_old + a1*g
v = (1-a2)*v_old + a2*g**2
w_new = (1-alpha*k)*w - alpha*m/(sqrt(v)+1e-8)
h_new = gamma*(1-alpha*k)*h + alpha*g/(sqrt(v)+1e-8)
M_new = r*M_old + (1-r)*z
beta_new = beta + eta*sign(c*M_new + (1-c)*z)
```

`m,v` are directly stored bias-corrected moments, equivalent to ordinary
EMA moments divided by `1-b**t` for dense active steps. The trace injects the
normalized **gradient**, not the first moment or the weight-decay update.
The block's old alpha is used throughout the current weight/trace update.
Lion uses the **new** meta moment: its decision equals
`sign(c*r*M_old + (1-c*r)*z)`. Thus `c_eff=c*r`, not `c`, relative to the
old-moment Lion convention. The positive trace pairs with a positive beta
increment. No meta weight decay is offered.

Groups: `scalar` (all tensors), `layerwise` (one per tensor, not semantic
module), or an explicit positive integer size list covering the parameter
sequence once. Membership must not shift when a gradient is absent. An
inactive parameter does not receive an update; an entirely inactive block
does not advance its meta state. Sparse gradients are unsupported.

## J: JAX EMA-corrected / new-meta-moment / new-alpha

One scalar beta covers the whole PyTree. Standard EMA moments are stored:

```
mu = b1*mu_old + (1-b1)*g; nu = b2*nu_old + (1-b2)*g**2
m_hat = mu/(1-b1**t); v_hat = nu/(1-b2**t)
z = sum(vdot(h_leaf, g_leaf))
M_new = r*M_old + (1-r)*z
beta_new = beta + eta*sign(c*M_new + (1-c)*z)
a = exp(beta_new)
update = -a*(m_hat/(sqrt(v_hat)+1e-8) + k_leaf*w)
h_new = gamma*(1-a*k_leaf)*h + a*g/(sqrt(v_hat)+1e-8)
```

Unlike P, J immediately uses **new** alpha in weights and trace. Its Lion
coefficient is also `c_eff=c*r`. The mask sets `k_leaf` to `k` for selected
leaves and zero otherwise, consistently in **both** decay and trace
contraction; it does not suppress the fresh gradient injection. Mask inputs
may be a matching boolean PyTree or a callable producing that tree; no mask
means decay all leaves. Parameters are required for decay. This mask/trace
consistency is a bug fix, not preservation of the old inconsistent recurrence.
Here `r=meta_b1`, `c=meta_c1`, and `t` is the global transformation update
count. Dense matching real-valued gradient/parameter PyTrees are the supported
contract; no per-leaf beta groups or PyTorch-style missing-gradient semantics
are promised. Apply returned updates with `optax.apply_updates`.

## E: experimental HF unnormalized-moment / old-moment / old-alpha

The CIFAR-10, ImageNet and TinyStories HF families preserve the following
active-gradient equations (config variant
`historical-unnormalized-moment-old-lion-negative-step-trace`). `alpha=exp(beta_old)`; compute `z=<h_old,g>` by
block, then perform the base update, then the meta update. Let `delta` be the
entire subtracted weight step, **including decay**:

```
w_new = w - delta
h_new = gamma*(1-alpha*k)*h_old - delta
```

This is a negative actual-step trace, unlike P/J's positive normalized-gradient
trace. Decay is unmasked for HF tensors with active gradients. Every base
rule below uses `delta=alpha*(direction+k*w)`:

| Base name | Direction and state timing |
| --- | --- |
| SGD | `g` |
| SGDm | `m_old`; then `m_new=b1*m_old+(1-b1)*g` (first step can be zero with no decay) |
| RMSProp | `g/sqrt(q*v_new+1e-10)`; `v_new=b2*v_old+g**2`, `q=(1-b2)/(1-b2**t)` |
| AdamW | `m_new/sqrt(q*v_new+1e-10)`; `m_new=b1*m_old+g`, same `v_new,q` as RMSProp |
| Lion | `sign(c*m_old+(1-c)*g)`; then `m_new=b1*m_old+(1-b1)*g` |

In particular E-AdamW's first moment has **neither `(1-b1)` normalization nor
first-moment bias correction**. At constant gradient 1, no decay, fixed
alpha .1, starting weight 1, its first two weights are approximately .9 and
.71, versus standard AdamW's .9 and .8. Relative to standard AdamW's momentum
contribution the factor is `(1-b1**t)/(1-b1)`, not a constant during startup.
E's epsilon is **inside** the square root, unlike P/J.

Base `b1=momentum_param`, `b2=normalizer_param`, and `c=Lion_beta2`.
Meta rules operate on `z` with meta decay `K` and step size `eta`; their
`r1=momentum_param`, `r2=normalizer_param`, and `c=Lion_beta2`:

| Meta name | Beta rule |
| --- | --- |
| fixed | unchanged |
| RMSProp | `beta_new=(1-eta*K)*beta-eta*z/sqrt(q*V_new+1e-10)`; `V_new=r2*V_old+z**2`, `q=(1-r2)/(1-r2**t)` |
| Adam | `beta_new=(1-eta*K)*beta-eta*M_new/sqrt(q*V_new+1e-10)`; `M_new=r1*M_old+z`, same second moment as above |
| Lion | `beta_new=(1-eta*K)*beta-eta*sign(c*M_old+(1-c)*z)`; then `M_new=r1*M_old+(1-r1)*z` |

Meta Adam retains the unnormalized first-moment convention too. Meta Lion uses
**old** momentum (`c_eff=c`), with a negative beta increment appropriate to
its negative trace. `K` shrinks beta, not alpha directly.

Groups: `scalar`, `layerwise` (one parameter tensor per block), or explicitly
validated blocks covering parameter tensors exactly once, represented by
positive sizes or disjoint parameter-name lists. The historical ResNet block
presets are architecture-dependent (18: `[3,12,15,15,15,2]`; 50:
`[3,30,39,57,30,2]`). Do not assume these presets fit a different model.
TinyStories also retains `transformer_blocks_15m` with tensor counts
`[1,9,9,9,9,9,10]`, not semantic transformer-block discovery.
`nodewise` and `weightwise` are not supported. Missing gradients freeze that
parameter's weights, decay, moments and trace. Entirely inactive blocks freeze
their meta state. Base bias-correction clocks count updates with **any** active
gradient; meta clocks count updates with any active block, not separate clocks
per tensor/block. This differs from P's per-parameter active-step clocks.

The shared experiment protocol consumes `.grad`: `zero_grad(set_to_none=True)`,
backward/accumulate, unscale/clip if used, then `step(model)` exactly once.
`step(model, loss)` is a convenience that clears gradients and calls backward
once. Do not pass a loss after accumulating gradients. Sparse gradients are
unsupported; absent gradients do not cause a different tensor to inherit that
tensor's block. Experiment `param_groups` permits gradient inspection but does
not promise native `torch.optim.Optimizer` inheritance.

## Native baselines and schedules (not M01/M02)

Native AdamW baselines use PyTorch's ordinary normalized/bias-corrected rule,
not E-AdamW. Their schedule is part of the effective configuration. Without a
supplied horizon the approved fallback is constant LR with
`scheduler="constant", warmup_steps=0, total_steps=null, min_lr=0`.
Nonzero warmup/minimum LR without a horizon is invalid. With a horizon, use
the actual configured warmup/total steps; TinyStories defaults its horizon to
`max_iters`. The historical hidden 10000/422000 bounds are not defaults to
reproduce. This is an explicit schedule correction distinct from preservation
of the HF equations above. Compare variants using recorded rates/schedules,
not merely identical CLI names.
