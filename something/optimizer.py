import torch


#@torch.compile
def zeroth_power_via_newtonschulz2(G, steps=9, eps=1e-7):
    
    assert len(G.shape) == 2
    X = G.bfloat16() / (torch.linalg.norm(G, ord='fro') + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        print("Hello")
        X = 2 * X - 1.5 * B + 0.5 * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class SpectralSGDM(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                state['steps'] = state.get('steps', 0) + 1
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                update = zeroth_power_via_newtonschulz2(g)
                p.data.add_(update, alpha=-lr)

class CombinedOptimizer:

    def __init__(self, optimizers):
        assert all(len(opt.param_groups) == 1 for opt in optimizers)
        self.optimizers = optimizers
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]['lr'] for opt in self.optimizers]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]['lr'] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]