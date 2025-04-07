import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional
import math
import random

def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out

def Cayley_loop(X, W, tan_vec, t): # 
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(4):
        Y = X + t * torch.matmul(W, 0.5*(X+Y))

    return Y.T

def Cayley_full(X, W, t):

    Y = (torch.eye(W.shape[0],W.shape[1]).to(X.device) + 0.5*t* W)@X
    Y  = torch.linalg.solve((torch.eye(W.shape[0],W.shape[1]).to(X.device) - 0.5*t*W), Y)
    return Y.T

def norm(v, dim=1):
    #print(v.shape)
    assert len(v.size())==2
    return v.norm(p=2, dim=dim, keepdim=True)

def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim)
    return v/vnorm.add(eps), vnorm

def qr_retraction(tan_vec): # tan_vec, p-by-n, p <= n
    [p,n] = tan_vec.size()
    tan_vec = tan_vec.T
    q,r = torch.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q = q.T
    return q

__all__ = ['SGD', 'sgd']

class ProjectedSGD(Optimizer):
    r"""Modified stochastic gradient descent .

    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ProjectedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    @torch.no_grad()
    def project_grads(self,param,d_p):

        if hasattr(param,'correction'):   
            if hasattr(param,'U') and param.U:
                d_p = (torch.eye(param.shape[0], device=param.device)-param@param.T)@d_p@(param.correction)
            elif hasattr(param,'V') and param.V:
                d_p = (torch.eye(param.shape[0], device=param.device)-param@param.T)@d_p@(param.correction)
        elif hasattr(param,'a'): 
            if hasattr(param,'U') and param.U:
                eye_p = torch.eye(param.shape[0], device=param.device)
                d_p = (eye_p-param@param.T)@d_p/(param.a) 
            elif hasattr(param,'V') and param.V:
                eye_p = torch.eye(param.shape[0], device=param.device)
                d_p = (eye_p-param@param.T)@d_p/(param.a) 
        return d_p

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            weight_decay=group['weight_decay']
            momentum=group['momentum']
            lr=group['lr']
            dampening=group['dampening']
            nesterov=group['nesterov']
            maximize=group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad if not maximize else -p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(d_p)#.detach()
                        buf = state['momentum_buffer']
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                d_p = self.project_grads(p,d_p)
                if momentum != 0:
                    state['momentum_buffer'] = torch.clone(d_p)
                p.data.add_(d_p, alpha=-lr)
        return loss
    

    def step_cayley_sgd(self, closure=None, retraction='cayley'):
        # from https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform for comparison
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            weight_decay=group['weight_decay']
            momentum=group['momentum']
            lr=group['lr']
            dampening=group['dampening']
            nesterov=group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                unity,_ = unit(p.data.view(p.size()[0],-1))
                if unity.size()[0] <= unity.size()[1]:
                    d_p = p.grad.data.view(p.size()[0],-1)

                    rand_num = random.randint(1,101)
                    if rand_num==1:
                        unity = qr_retraction(unity)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] =  torch.zeros(d_p.T.size())
                        if p.is_cuda:
                            state['momentum_buffer'] = state['momentum_buffer'].to(p.device)
                    
                    buf = state['momentum_buffer'] 
                    buf = momentum * buf - d_p.T
                    bs = torch.mm(buf, unity)
                    sbs = torch.mm(unity, bs)
                    ssbs = torch.mm(unity.T, sbs)
                    w_h = bs-0.5*ssbs
                    w = w_h - w_h.T
                    epsilon = 1e-8
                    lr_ad = 0.5 * 2 / (matrix_norm_one(w) + epsilon)   
                    alpha = min(lr_ad, lr)
                    if retraction=='cayley':   
                        p_new = Cayley_loop(unity.T, w, buf, alpha)
                    elif retraction=='cayley_full':
                        p_new = Cayley_full(unity.T, w, alpha)
                    elif retraction == 'qr':
                        #print()
                        p_new = qr_retraction(unity+alpha*unity@w.T)
                    buf_new = torch.mm(w, unity.T) 
                    p.data.copy_(p_new.view(p.size()))
                    buf.copy_(buf_new)
                else:
                    d_p = p.grad.data 

                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    if momentum != 0:

                        state = self.state[p]
                        if 'momentum_buffer' not in state:
                            state['momentum_buffer'] = d_p.clone()#.detach()
                            buf = state['momentum_buffer']
                        else:
                            buf = state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)

                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    p.data.add_(-lr, d_p)
                    
        return loss
