"""Historical experiment HF optimizer, using synchronized/accumulated ``p.grad``.

AdamW and meta-Adam intentionally accumulate an *unnormalized* first moment
(no first-moment bias correction). Second moments are bias corrected, with
1e-10 inside the square root. Lion uses the old moment for its sign update,
then updates that moment. Base SGDm likewise steps with its previous moment.
The sensitivity trace injects the negative actual
parameter step, including decay. These are historical variants, not torch AdamW.
Missing gradients leave the parameter, its moments and sensitivity unchanged;
fully inactive blocks also leave their meta state unchanged. Bias-correction
clocks count optimizer steps with any active gradient (not per-parameter steps).
Sparse gradients are unsupported and rejected before any update.
"""
import ast
import copy
import math
import torch


class HF:
    def __init__(self, net, stepsize_groups, alpha0, args_base, args_meta, gamma, writer=None):
        self.params = list(net.parameters())
        self.names = [name.removeprefix('module.') for name, _ in net.named_parameters()]
        if not self.params:
            raise ValueError('HF requires at least one parameter')
        if not math.isfinite(alpha0) or alpha0 <= 0:
            raise ValueError('alpha0 must be finite and positive')
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError('gamma must be in [0, 1]')
        self.args_base = dict(args_base)
        self.args_meta = dict(args_meta)
        required_base = {
            'SGD': ['weight_decay'], 'SGDm': ['weight_decay', 'momentum_param'],
            'RMSProp': ['weight_decay', 'normalizer_param'],
            'AdamW': ['weight_decay', 'normalizer_param', 'momentum_param'],
            'Lion': ['weight_decay', 'momentum_param', 'Lion_beta2']}
        required_meta = {
            'fixed': [], 'RMSProp': ['meta_stepsize', 'weight_decay', 'normalizer_param'],
            'Adam': ['meta_stepsize', 'weight_decay', 'normalizer_param', 'momentum_param'],
            'Lion': ['meta_stepsize', 'weight_decay', 'momentum_param', 'Lion_beta2']}
        for label, args, required in [('base', self.args_base, required_base),
                                      ('meta', self.args_meta, required_meta)]:
            alg = args.get('alg')
            if alg not in required:
                raise ValueError(f'unsupported {label} algorithm: {alg!r}')
            missing = set(required[alg]) - args.keys()
            if missing:
                raise ValueError(f'missing {label} arguments: {sorted(missing)}')
            for key in required[alg]:
                value = args[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{label} {key} must be finite')
                if key in ('momentum_param', 'normalizer_param', 'Lion_beta2'):
                    if not 0 <= value < 1:
                        raise ValueError(f'{label} {key} must be in [0, 1)')
                elif value < 0:
                    raise ValueError(f'{label} {key} must be nonnegative')
        self.gamma, self.writer = gamma, writer
        self.num_layers = len(self.params)
        self.param_groups_indices = self._groups(stepsize_groups)
        self.stepsize_type = 'scalar' if stepsize_groups == 'scalar' else 'blockwise'
        self.num_blocks = len(self.param_groups_indices)
        self.map_layers_to_blocks = [next(j for j, group in enumerate(self.param_groups_indices) if i in group)
                                    for i in range(self.num_layers)]
        # Meta state is small and stays on the model device; no NumPy conversion.
        self.beta = [torch.full((self.num_blocks,), math.log(alpha0),
                                dtype=self.params[0].dtype, device=self.params[0].device)]
        self.h_condenced = [torch.zeros_like(p) for p in self.params]
        self.trace_base = [torch.zeros_like(p) for p in self.params]
        self.momentum_base = [torch.zeros_like(p) for p in self.params]
        self.trace_meta = [torch.zeros_like(self.beta[0])]
        self.momentum_meta = [torch.zeros_like(self.beta[0])]
        self.lambda_base_t = self.lambda_meta_t = 1.0
        self.epsilon = 1e-10
        self.counter = -1
        self.config = dict(optimizer='HF', variant='historical-unnormalized-moment-old-lion-negative-step-trace',
                           args_base=self.args_base, args_meta=self.args_meta, gamma=gamma,
                           alpha0=alpha0, groups=self.param_groups_indices)
        self.param_groups = [dict(params=[self.params[i] for i in group], lr=float(alpha0))
                             for group in self.param_groups_indices]
        self.alpha = [float(alpha0)] * self.num_layers

    def _groups(self, groups):
        """Scalar, layerwise, exact named partitions or positive block sizes only."""
        if isinstance(groups, str):
            if groups == 'scalar':
                return [list(range(self.num_layers))]
            if groups == 'layerwise':
                return [[i] for i in range(self.num_layers)]
            named = {'resnet18_blocks': [3, 12, 15, 15, 15, 2],
                     'resnet50_blocks': [3, 30, 39, 57, 30, 2],
                     'transformer_blocks_15m': [1, 9, 9, 9, 9, 9, 10]}
            if groups in named:
                groups = named[groups]
            else:
                try:
                    groups = ast.literal_eval(groups)
                except (ValueError, SyntaxError):
                    raise ValueError('unsupported stepsize groups; use scalar, layerwise, resnet18_blocks, '
                                     'resnet50_blocks, transformer_blocks_15m or an explicit partition') from None
        if not isinstance(groups, list) or not groups:
            raise ValueError('stepsize groups must be a nonempty list')
        if all(type(size) is int for size in groups):
            if any(size <= 0 for size in groups) or sum(groups) != self.num_layers:
                raise ValueError('positive block sizes must sum to the number of parameter tensors')
            result, start = [], 0
            for size in groups:
                result.append(list(range(start, start + size)))
                start += size
            return result
        if not all(isinstance(group, list) and group and all(isinstance(n, str) for n in group)
                   for group in groups):
            raise ValueError('named groups must be nonempty lists of parameter names')
        flat = [name for group in groups for name in group]
        if len(flat) != len(set(flat)) or set(flat) != set(self.names):
            raise ValueError('named groups must partition every parameter exactly once')
        return [[self.names.index(name) for name in group] for group in groups]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def get_learning_rates(self):
        return [float(a) for a in self.beta[0].detach().exp().cpu().tolist()]

    @torch.no_grad()
    def _update(self):
        gradients = [p.grad for p in self.params]
        if any(g is not None and g.is_sparse for g in gradients):
            raise ValueError('HF does not support sparse gradients')
        active = [any(gradients[i] is not None for i in group) for group in self.param_groups_indices]
        if not any(active):
            return
        rates = self.get_learning_rates()
        self.alpha = [rates[j] for j in self.map_layers_to_blocks]
        hypergrad = torch.stack([sum(((self.h_condenced[i] * gradients[i]).sum()
                                     for i in group if gradients[i] is not None),
                                    self.beta[0].new_zeros(()))
                                 for group in self.param_groups_indices])
        base = self.args_base
        alg = base['alg']
        if alg in ('RMSProp', 'AdamW'):
            self.lambda_base_t *= base['normalizer_param']
            mu = (1 - base['normalizer_param']) / (1 - self.lambda_base_t)
        for i, (p, g, a) in enumerate(zip(self.params, gradients, self.alpha)):
            if g is None:
                continue
            moment = self.momentum_base[i]
            if alg == 'SGD':
                direction = g
            elif alg == 'SGDm':
                direction = moment
                self.momentum_base[i] = base['momentum_param'] * moment + (1 - base['momentum_param']) * g
            elif alg == 'Lion':
                direction = torch.sign(base['Lion_beta2'] * moment + (1 - base['Lion_beta2']) * g)
                self.momentum_base[i] = base['momentum_param'] * moment + (1 - base['momentum_param']) * g
            else:
                self.trace_base[i] = base['normalizer_param'] * self.trace_base[i] + g.square()
                if alg == 'AdamW':
                    self.momentum_base[i] = base['momentum_param'] * moment + g
                    direction = self.momentum_base[i]
                else:
                    direction = g
                direction = direction / (mu * self.trace_base[i] + self.epsilon).sqrt()
            delta = a * (direction + base['weight_decay'] * p)
            p.sub_(delta)
            self.h_condenced[i] = self.gamma * (1 - base['weight_decay'] * a) * self.h_condenced[i] - delta
        meta = self.args_meta
        alg = meta['alg']
        if alg != 'fixed':
            mask = torch.tensor(active, device=self.beta[0].device)
            old_m = self.momentum_meta[0]
            if alg in ('RMSProp', 'Adam'):
                self.lambda_meta_t *= meta['normalizer_param']
                mu = (1 - meta['normalizer_param']) / (1 - self.lambda_meta_t)
                trace = meta['normalizer_param'] * self.trace_meta[0] + hypergrad.square()
                self.trace_meta[0] = torch.where(mask, trace, self.trace_meta[0])
                if alg == 'Adam':
                    moment = meta['momentum_param'] * old_m + hypergrad
                    self.momentum_meta[0] = torch.where(mask, moment, old_m)
                    direction = moment
                else:
                    direction = hypergrad
                direction = direction / (mu * trace + self.epsilon).sqrt()
            else:
                direction = torch.sign(meta['Lion_beta2'] * old_m + (1 - meta['Lion_beta2']) * hypergrad)
                self.momentum_meta[0] = torch.where(mask, meta['momentum_param'] * old_m +
                                                   (1 - meta['momentum_param']) * hypergrad, old_m)
            beta = (1 - meta['meta_stepsize'] * meta['weight_decay']) * self.beta[0] - meta['meta_stepsize'] * direction
            self.beta[0] = torch.where(mask, beta, self.beta[0])
        for group, rate in zip(self.param_groups, self.get_learning_rates()):
            group['lr'] = rate
        self.counter += 1
        if self.writer is not None and self.counter % 100 == 0:
            for i, rate in enumerate(self.get_learning_rates()):
                suffix = 'scalar' if self.stepsize_type == 'scalar' else 'block' + str(i)
                prefix = 'Optimizer_' + self.stepsize_type + '/'
                self.writer.add_scalar(prefix + 'alpha_' + suffix, rate, self.counter)
                self.writer.add_scalar(prefix + 'beta_' + suffix, float(self.beta[0][i]), self.counter)

    def step(self, model, loss=None):
        if [id(p) for p in model.parameters()] != [id(p) for p in self.params]:
            raise ValueError('HF.step model parameters differ from optimizer parameters')
        if loss is not None:
            self.zero_grad()
            loss.backward()
        self._update()

    _state_fields = ('beta', 'h_condenced', 'trace_base', 'momentum_base', 'trace_meta',
                     'momentum_meta', 'lambda_base_t', 'lambda_meta_t', 'counter')

    def state_dict(self):
        return copy.deepcopy(dict(version=1, config=self.config, names=self.names,
                                  shapes=[list(p.shape) for p in self.params],
                                  state={key: getattr(self, key) for key in self._state_fields}))

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get('version') != 1 or state.get('config') != self.config or
                state.get('names') != self.names or state.get('shapes') != [list(p.shape) for p in self.params]):
            raise ValueError('incompatible HF state configuration or parameters')
        values = state.get('state', {})
        if not isinstance(values, dict) or set(values) != set(self._state_fields):
            raise ValueError('incomplete HF state')
        loaded = {}
        second_moments = set()
        if self.args_base['alg'] in ('RMSProp', 'AdamW'):
            second_moments.add('trace_base')
        if self.args_meta['alg'] in ('RMSProp', 'Adam'):
            second_moments.add('trace_meta')
        for key in self._state_fields:
            value, expected = values[key], getattr(self, key)
            if isinstance(expected, list):
                if (not isinstance(value, list) or len(value) != len(expected) or
                        any(not isinstance(v, torch.Tensor) or v.shape != e.shape for v, e in zip(value, expected))):
                    raise ValueError(f'incompatible HF state tensor shapes: {key}')
                if any(v.layout != torch.strided or not v.is_floating_point() for v in value):
                    raise ValueError(f'invalid HF state tensor layout or dtype: {key}')
                loaded[key] = [v.detach().to(device=e.device, dtype=e.dtype).clone() for v, e in zip(value, expected)]
                if any(not torch.isfinite(v).all() or
                       (key in second_moments and (v < 0).any()) for v in loaded[key]):
                    raise ValueError(f'invalid HF state tensor values: {key}')
            else:
                if not isinstance(value, type(expected)) or not math.isfinite(value):
                    raise ValueError(f'invalid HF state: {key}')
                if key in ('lambda_base_t', 'lambda_meta_t') and not 0 <= value <= 1:
                    raise ValueError(f'invalid HF bias-correction product: {key}')
                loaded[key] = value
        rates = [float(rate) for rate in loaded['beta'][0].exp().cpu().tolist()]
        if not all(math.isfinite(rate) for rate in rates):
            raise ValueError('invalid HF state learning rates')
        for key, value in loaded.items():
            setattr(self, key, value)
        for group, rate in zip(self.param_groups, rates):
            group['lr'] = rate
