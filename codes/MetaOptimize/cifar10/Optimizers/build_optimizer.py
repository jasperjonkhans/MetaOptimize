"""Experiment optimizer builders and the shared backward/precomputed-grad protocol."""
import copy
import math
import torch
from torch.optim.lr_scheduler import StepLR


class CosineDecayWithWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup then cosine decay, clamped at min_lr after total_steps."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        if (type(warmup_steps) is not int or type(total_steps) is not int or
                not 0 <= warmup_steps < total_steps):
            raise ValueError('schedule requires integer 0 <= warmup_steps < total_steps')
        if not math.isfinite(min_lr) or min_lr < 0 or any(min_lr > g['lr'] for g in optimizer.param_groups):
            raise ValueError('min_lr must be finite and between zero and the initial learning rate')
        self.warmup_steps, self.total_steps, self.min_lr = warmup_steps, total_steps, min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(max(self.last_epoch, 0), self.total_steps)
        if step < self.warmup_steps:
            return [base * step / self.warmup_steps for base in self.base_lrs]
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [self.min_lr + (base - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base in self.base_lrs]


def _schedule(optimizer, warmup_steps, total_steps, min_lr):
    if total_steps is None:
        if warmup_steps != 0 or min_lr != 0:
            raise ValueError('warmup_steps/min_lr require total_steps')
        return None
    return CosineDecayWithWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr)


class _NativeWrapper:
    def __init__(self, optimizer, config, scheduler=None):
        self.optimizer, self.scheduler = optimizer, scheduler
        self.config = dict(config, scheduler=('constant' if scheduler is None else
                                             'cosine' if isinstance(scheduler, CosineDecayWithWarmupScheduler)
                                             else type(scheduler).__name__))

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, model, loss=None):
        params = [p for group in self.param_groups for p in group['params']]
        if [id(p) for p in model.parameters()] != [id(p) for p in params]:
            raise ValueError('step model parameters differ from optimizer parameters')
        if loss is not None:
            self.zero_grad()
            loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def get_learning_rates(self):
        return [float(group['lr']) for group in self.param_groups]

    def state_dict(self):
        return copy.deepcopy(dict(version=1, config=self.config,
                                  shapes=[[list(p.shape) for p in g['params']] for g in self.param_groups],
                                  optimizer=self.optimizer.state_dict(),
                                  scheduler=None if self.scheduler is None else self.scheduler.state_dict()))

    def load_state_dict(self, state):
        shapes = [[list(p.shape) for p in g['params']] for g in self.param_groups]
        if (not isinstance(state, dict) or state.get('version') != 1 or state.get('config') != self.config or
                state.get('shapes') != shapes or 'optimizer' not in state or
                'scheduler' not in state or (state['scheduler'] is None) != (self.scheduler is None)):
            raise ValueError('incompatible native optimizer state configuration or parameters')
        previous = self.state_dict()
        try:
            self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            if self.scheduler is not None:
                self.scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
        except (ValueError, KeyError, TypeError) as error:
            self.optimizer.load_state_dict(previous['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(previous['scheduler'])
            raise ValueError('invalid native optimizer/scheduler state') from error


class SGD_optimizer(_NativeWrapper):
    def __init__(self, net, lr, writer=None, weight_decay=0.0):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
        super().__init__(optimizer, dict(optimizer='SGD', variant='torch-SGD', lr=lr,
                                        weight_decay=weight_decay, step_size=15000, scheduler_gamma=0.1),
                         StepLR(optimizer, step_size=15000, gamma=0.1))


class SGDm_optimizer(_NativeWrapper):
    def __init__(self, net, lr, momentum, writer=None, weight_decay=0.0):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(optimizer, dict(optimizer='SGDm', variant='torch-SGD-momentum', lr=lr,
                                        momentum=momentum, weight_decay=weight_decay,
                                        step_size=15000, scheduler_gamma=0.1),
                         StepLR(optimizer, step_size=15000, gamma=0.1))


class Adam_optimizer(_NativeWrapper):
    """Preserves the historical 'Adam' CLI alias for native torch AdamW."""
    def __init__(self, net, lr, data=None, writer=None, weight_decay=0.01, betas=(0.9, 0.999)):
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(optimizer, dict(optimizer='Adam', variant='historical-alias-torch-AdamW',
                                        lr=lr, weight_decay=weight_decay, betas=list(betas), eps=1e-8))


class AdamW_optimizer(_NativeWrapper):
    def __init__(self, net, lr, data=None, writer=None, weight_decay=0.1, betas=(0.9, 0.999),
                 warmup_steps=0, total_steps=None, min_lr=0.0):
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scheduler = _schedule(optimizer, warmup_steps, total_steps, min_lr)
        super().__init__(optimizer, dict(optimizer='AdamW', variant='torch-AdamW', lr=lr,
                                        weight_decay=weight_decay, betas=list(betas), eps=1e-8,
                                        warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr,
                                        scheduler='constant' if scheduler is None else 'cosine'), scheduler)


class _HistoricalLion(torch.optim.Optimizer):
    """Sign of old moment mixed with gradient, then update the moment."""
    def __init__(self, params, lr, weight_decay, momentum, sign_momentum):
        if (not math.isfinite(lr) or lr < 0 or not math.isfinite(weight_decay) or weight_decay < 0 or
                not 0 <= momentum < 1 or not 0 <= sign_momentum < 1):
            raise ValueError('invalid Lion learning rate, decay or moments')
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                     momentum=momentum, sign_momentum=sign_momentum))

    @torch.no_grad()
    def step(self, closure=None):
        if any(p.grad is not None and p.grad.is_sparse for group in self.param_groups for p in group['params']):
            raise ValueError('Lion does not support sparse gradients')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                moment = state['momentum']
                direction = (group['sign_momentum'] * moment + (1 - group['sign_momentum']) * p.grad).sign()
                p.add_(direction + group['weight_decay'] * p, alpha=-group['lr'])
                moment.mul_(group['momentum']).add_(p.grad, alpha=1 - group['momentum'])


class Lion_optimizer(_NativeWrapper):
    def __init__(self, net, lr, weight_decay, writer=None, momentum=0.99, sign_momentum=0.9,
                 warmup_steps=0, total_steps=None, min_lr=0.0):
        optimizer = _HistoricalLion(net.parameters(), lr, weight_decay, momentum, sign_momentum)
        scheduler = _schedule(optimizer, warmup_steps, total_steps, min_lr)
        super().__init__(optimizer, dict(optimizer='Lion', variant='historical-old-moment-Lion',
                                        lr=lr, weight_decay=weight_decay, momentum=momentum,
                                        sign_momentum=sign_momentum, warmup_steps=warmup_steps,
                                        total_steps=total_steps, min_lr=min_lr), scheduler)


def build_optimizer(net, args, writer=None):
    if args.optimizer == 'HF':
        from Optimizers.HF import HF
        fields = ['alg', 'normalizer_param', 'momentum_param', 'weight_decay', 'Lion_beta2']
        base = {key: getattr(args, key + '_base') for key in fields if getattr(args, key + '_base', None) is not None}
        meta = {key: getattr(args, key + '_meta') for key in fields if getattr(args, key + '_meta', None) is not None}
        meta['meta_stepsize'] = args.meta_stepsize
        return HF(net, args.stepsize_groups, args.alpha0, base, meta, args.gamma, writer)
    if args.optimizer not in ('AdamW',):
        raise ValueError(f'unsupported optimizer: {args.optimizer!r}')
    if args.optimizer in ('SGD', 'SGDm', 'Adam'):
        lr = args.lr
        decay = getattr(args, 'weight_decay', 0.0)
        if args.optimizer == 'SGD':
            return SGD_optimizer(net, lr, writer, decay)
        if args.optimizer == 'SGDm':
            return SGDm_optimizer(net, lr, args.momentum, writer, decay)
        return Adam_optimizer(net, lr, writer=writer, weight_decay=decay,
                              betas=(getattr(args, 'momentum_param_base', 0.9),
                                     getattr(args, 'normalizer_param_base', 0.999)))
    total = getattr(args, 'total_steps', None)
    schedule = dict(warmup_steps=getattr(args, 'warmup_steps', 0), total_steps=total,
                    min_lr=getattr(args, 'min_lr', 0.0))
    if args.optimizer == 'Lion':
        return Lion_optimizer(net, args.alpha0, args.weight_decay_base, writer,
                              momentum=getattr(args, 'momentum_param_base', 0.99),
                              sign_momentum=getattr(args, 'Lion_beta2_base', 0.9), **schedule)
    return AdamW_optimizer(net, args.alpha0, writer=writer,
                           weight_decay=getattr(args, 'weight_decay_base', 0.1),
                           betas=(getattr(args, 'momentum_param_base', 0.9),
                                  getattr(args, 'normalizer_param_base', 0.999)), **schedule)
