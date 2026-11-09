import math
from numbers import Integral, Real

import torch
import torch.nn as nn


class MetaOptimize_AdamW_Lion_hg:
    """Manual dense PyTorch optimizer, not a ``torch.optim.Optimizer``.

    Call backward(), then step(); step consumes p.grad without clearing it.
    zero_grad() clears existing gradients to zero (not None). Missing gradients
    skip weights, decay and per-parameter state. Fixed blocks still finish a meta
    update when partially active; fully inactive blocks freeze all their state.

    Variant ``package_adamw_lion_hg_v1`` stores bias-corrected AdamW moments:
    m_hat_new = (1-q1)*m_hat_old + q1*g, q1 = (1-b1)/(1-b1**step)
    (and analogously v_hat with b2 and g**2). Weights use the OLD block rate
    alpha = exp(beta), with decoupled decay and denominator sqrt(v_hat)+eps.
    For z = sum(h_old*g) over active members, Lion uses the NEW meta momentum:
    M_new = meta_b1*M_old + (1-meta_b1)*z,
    beta_new = beta_old + meta_stepsize*sign(meta_c1*M_new + (1-meta_c1)*z).
    The trace is h_new = gamma*(1-alpha*decay)*h_old
    + alpha*g/(sqrt(v_hat)+eps), not the full momentum/decay weight step.
    Beta retains historical float32 precision; meta momentum follows gradient
    accumulation precision. No native schedulers, parameter groups, closures
    or automatic backward are supported.
    """

    _VARIANT = 'package_adamw_lion_hg_v1'
    _CONFIG_KEYS = ('meta_stepsize', 'alpha0', 'gamma', 'b1', 'b2',
                    'meta_b1', 'meta_c1', 'weight_decay', 'eps')

    @staticmethod
    def _validate_config(config):
        for name in MetaOptimize_AdamW_Lion_hg._CONFIG_KEYS:
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite real number')
            if name in ('alpha0', 'eps'):
                valid = value > 0
            elif name in ('meta_stepsize', 'weight_decay'):
                valid = value >= 0
            elif name == 'gamma':
                valid = 0 <= value <= 1
            else:
                valid = 0 <= value < 1
            if not valid:
                raise ValueError(f'{name} is outside its supported range')

    @staticmethod
    def _validate_blocks(blocks, count):
        if (not isinstance(blocks, (list, tuple)) or not blocks or
                any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in blocks) or
                sum(blocks) != count):
            raise ValueError('block sizes must be positive integers summing to the parameter count')
        return tuple(int(n) for n in blocks)

    def __init__(self, params, meta_stepsize=1e-3, alpha0=1e-3,
                 stepsize_blocks='scalar', gamma=0.9999, b1=0.9, b2=0.999,
                 meta_b1=0.99, meta_c1=0.9, weight_decay=0.0):
        params = list(params)
        if not params:
            raise ValueError('params must be nonempty')
        if any(not isinstance(p, torch.Tensor) or not p.is_floating_point() or
               p.layout != torch.strided for p in params):
            raise ValueError('params must be dense floating-point tensors')
        if len({id(p) for p in params}) != len(params):
            raise ValueError('params must not contain duplicate tensors')
        config = dict(meta_stepsize=meta_stepsize, alpha0=alpha0, gamma=gamma,
                      b1=b1, b2=b2, meta_b1=meta_b1, meta_c1=meta_c1,
                      weight_decay=weight_decay, eps=1e-8)
        self._validate_config(config)
        if isinstance(stepsize_blocks, str):
            if stepsize_blocks == 'scalar':
                blocks = [len(params)]
            elif stepsize_blocks == 'layerwise':
                blocks = [1] * len(params)
            else:
                raise ValueError('stepsize_blocks must be scalar, layerwise, or a list of sizes')
        else:
            blocks = stepsize_blocks
        blocks = self._validate_blocks(blocks, len(params))
        # All validation precedes state allocation; no parameter is mutated here.
        self.params = params
        self._block_sizes = blocks
        for name, value in config.items():
            setattr(self, name, float(value))
        self.state = {p: dict(step=0, base_momentum=torch.zeros_like(p),
                              base_normalizer=torch.zeros_like(p), h=torch.zeros_like(p))
                      for p in params}
        self.meta_state = []
        offset = 0
        for size in blocks:
            device = params[offset].device
            self.meta_state.append(dict(num_layers_in_this_block=size,
                                        meta_momentum=torch.tensor(0., dtype=torch.float32, device=device),
                                        beta=torch.tensor(math.log(alpha0), dtype=torch.float32, device=device)))
            offset += size

    @torch.no_grad()
    def step(self):
        """Apply one update using existing dense gradients; reject sparse atomically."""
        if any(p.grad is not None and p.grad.layout != torch.strided for p in self.params):
            raise ValueError('sparse gradients are not supported')
        offset = 0
        for size, meta_state in zip(self._block_sizes, self.meta_state):
            members = self.params[offset:offset + size]
            offset += size
            active = [p for p in members if p.grad is not None]
            if not active:
                continue
            beta = meta_state['beta']
            alpha = torch.exp(beta)
            meta_grad = 0.0
            for p in active:
                grad = p.grad
                state = self.state[p]
                base_momentum = state['base_momentum']
                base_normalizer = state['base_normalizer']
                h = state['h']
                state['step'] += 1
                b1 = (1-self.b1)/(1-self.b1**state['step'])
                b2 = (1-self.b2)/(1-self.b2**state['step'])
                base_momentum.mul_(1-b1).add_(grad, alpha=b1)
                base_normalizer.mul_(1-b2).addcmul_(grad, grad, value=b2)
                if self.weight_decay > 0:
                    p.add_(p, alpha=-alpha*self.weight_decay)
                p.add_(base_momentum/(base_normalizer.sqrt()+self.eps), alpha=-alpha)
                meta_grad += (h * grad).sum().to(beta.device)
                h.mul_((self.gamma*(1-alpha*self.weight_decay)).item()).add_(
                    grad/(base_normalizer.sqrt()+self.eps), alpha=alpha)
            momentum = self.meta_b1 * meta_state['meta_momentum'] + (1-self.meta_b1) * meta_grad
            update = self.meta_c1 * momentum + (1-self.meta_c1) * meta_grad
            beta.add_(self.meta_stepsize * update.sign())
            meta_state['meta_momentum'] = momentum

    def zero_grad(self):
        """Clear existing gradients to zero; preserve missing gradients as None."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self):
        """Return an independent v1 optimizer snapshot (not model weights or RNG).

        Parameters are identified by their supplied order and shapes. Save model
        weights separately. This schema is not a native torch.optim state dict.
        """
        return dict(version=1, variant=self._VARIANT,
                    param_shapes=[tuple(p.shape) for p in self.params],
                    block_sizes=list(self._block_sizes),
                    hyperparameters={k: getattr(self, k) for k in self._CONFIG_KEYS},
                    state=[{k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                            for k, v in self.state[p].items()} for p in self.params],
                    meta_state=[{k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                                 for k, v in s.items()} for s in self.meta_state])

    @torch.no_grad()
    def load_state_dict(self, checkpoint):
        """Atomically restore v1 state AND hyperparameters, without input aliasing.

        Ordered shapes and block sizes must match. Base tensors follow destination
        parameter dtype/device; beta stays float32 and meta momentum retains its
        saved precision, both on each block's device.
        Same-shaped parameter reordering cannot be detected; callers must preserve
        order. Model weights, gradients and RNG are not loaded.
        """
        def fail(message):
            raise ValueError(f'invalid optimizer checkpoint: {message}')

        if not isinstance(checkpoint, dict) or set(checkpoint) != {
                'version', 'variant', 'param_shapes', 'block_sizes',
                'hyperparameters', 'state', 'meta_state'}:
            fail('schema')
        if type(checkpoint['version']) is not int or checkpoint['version'] != 1 or checkpoint['variant'] != self._VARIANT:
            fail('unsupported version or variant')
        if checkpoint['param_shapes'] != [tuple(p.shape) for p in self.params]:
            fail('parameter layout mismatch')
        blocks = self._validate_blocks(checkpoint['block_sizes'], len(self.params))
        if blocks != self._block_sizes:
            fail('block layout mismatch')
        config = checkpoint['hyperparameters']
        if not isinstance(config, dict) or set(config) != set(self._CONFIG_KEYS):
            fail('hyperparameters')
        self._validate_config(config)

        def tensor(value, shape, device, dtype):
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
                    not value.is_floating_point() or tuple(value.shape) != shape or
                    not torch.isfinite(value).all().item()):
                fail('state tensor shape, layout or value')
            result = value.detach().to(device=device, dtype=dtype).clone()
            if not torch.isfinite(result).all().item():
                fail('state tensor is not representable in destination dtype')
            return result

        saved = checkpoint['state']
        if not isinstance(saved, list) or len(saved) != len(self.params):
            fail('parameter state count')
        state = {}
        for p, entry in zip(self.params, saved):
            if not isinstance(entry, dict) or set(entry) != {'step', 'base_momentum', 'base_normalizer', 'h'}:
                fail('parameter state fields')
            if type(entry['step']) is not int or entry['step'] < 0:
                fail('step must be a nonnegative integer')
            state[p] = {'step': entry['step']}
            for key in ('base_momentum', 'base_normalizer', 'h'):
                state[p][key] = tensor(entry[key], tuple(p.shape), p.device, p.dtype)
            if (state[p]['base_normalizer'] < 0).any().item():
                fail('negative base normalizer')
        saved_meta = checkpoint['meta_state']
        if not isinstance(saved_meta, list) or len(saved_meta) != len(blocks):
            fail('meta state count')
        meta_state = []
        offset = 0
        for size, entry in zip(blocks, saved_meta):
            if (not isinstance(entry, dict) or set(entry) != {'num_layers_in_this_block', 'beta', 'meta_momentum'} or
                    type(entry['num_layers_in_this_block']) is not int or entry['num_layers_in_this_block'] != size):
                fail('meta state layout')
            device = self.params[offset].device
            meta_state.append(dict(num_layers_in_this_block=size,
                                   beta=tensor(entry['beta'], (), device, torch.float32),
                                   meta_momentum=tensor(entry['meta_momentum'], (), device,
                                                        entry['meta_momentum'].dtype if isinstance(entry['meta_momentum'], torch.Tensor) else torch.float32)))
            offset += size
        # Commit only after the entire checkpoint has passed validation.
        for name, value in config.items():
            setattr(self, name, float(value))
        self.state = state
        self.meta_state = meta_state


# -----------------------------------------------------------------------------

# --- Simple MLP model for MNIST ---
def get_mlp(num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

# --- Load MNIST data ---
def get_mnist_dataloaders():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# --- Evaluation function ---
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100. * correct / total



if __name__ == "__main__":
    # Metaparameters
    epochs = 50
    meta_stepsize = 1e-3
    alpha0 = 1e-3
    stepsize_blocks = 'scalar' # 'scalar', 'blockwise', or [2, 2, 2]
    b1, b2 = 0.9, 0.999
    meta_b1, meta_c1 = 0.99, 0.9
    weight_decay = 0.1
    gamma = 0.9999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data and model
    trainloader, testloader = get_mnist_dataloaders()
    model = get_mlp(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = MetaOptimize_AdamW_Lion_hg(params=model.parameters(),
                                           meta_stepsize=meta_stepsize,
                                           alpha0=alpha0,
                                           stepsize_blocks=stepsize_blocks,
                                           gamma=gamma,
                                           b1=b1, b2=b2,
                                           meta_b1=meta_b1,
                                           meta_c1=meta_c1,
                                           weight_decay=weight_decay)
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100. * correct / total
        test_loss, test_acc = evaluate(model, testloader, device, criterion)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
    
