import torch
from CondLR_utils.optimizer_LR.ProjectedSGD import ProjectedSGD
import math
from einops import rearrange
from ..layers.conv_usv import Conv2d_USV
from ..layers.linear_usv import Linear_USV



class opt_USV:
     
    def __init__(self,NN,baseline = False,**kwargs):
        self.NN = NN
        self.baseline = baseline
        self.kw = dict(**kwargs)
        self.retraction_opt = self.kw['retraction_opt']
        del self.kw['retraction_opt']
        self.stiefel_opt = self.kw['stiefel_opt']
        del self.kw['stiefel_opt']
        self.eps = self.kw['eps'] if 'eps' in self.kw.keys() else 0.1
        del self.kw['eps']
        if self.baseline and not self.stiefel_opt == 'cayley_sgd':
            self.integrator = torch.optim.SGD(NN.parameters(),**self.kw)
            # self.integrator = torch.optim.Adam(NN.parameters())
        else:
            self.integrator = ProjectedSGD(NN.parameters(),**self.kw)

    @torch.no_grad()
    def step(self):
        self.integrator.step()
        self.soft_qr()

    @torch.no_grad()
    def step_a(self):
        self.integrator.step()
        for l in self.NN.lr_model:
            if hasattr(l,'lr') and l.lr:
                l.a = float((1./l.S_hat.shape[0])*torch.trace(l.S_hat@l.S_hat.T).detach())
        self.retract()
            


    @torch.no_grad()
    def usv_step(self):
        if self.stiefel_opt == 'cayley_sgd':
            self.integrator.step_cayley_sgd(retraction=self.retraction_opt)
        elif self.baseline:
            self.integrator.step()
        elif self.stiefel_opt == 'mean':
            self.step_a()
        elif self.stiefel_opt == 'approx_orth':
            self.step()
        else:
            print("NOT IN METHODS, DOING THE MOST BASIC BASELINE")
            self.integrator.step()

    @torch.no_grad()
    def zero_grad(self):
        for p in self.NN.parameters():
            p.grad = None


    @torch.no_grad()
    def retract(self):
        for l in self.NN.lr_model:
            
            if hasattr(l,'lr') and l.lr:

                q_pru, r_pru = torch.linalg.qr(l.U.data)
                d = torch.diag(r_pru, 0)
                ph = d.sign()
                q_pru *= ph.expand_as(q_pru)
                l.U.data = q_pru

                q_prv, r_prv = torch.linalg.qr(l.V.data)
                d = torch.diag(r_prv, 0)
                ph = d.sign()
                q_prv *= ph.expand_as(q_prv)
                l.V.data = q_prv

                u,rq = torch.linalg.qr(l.S_hat.data)
                d = torch.diag(rq, 0)
                ph = d.sign()
                u *= ph.expand_as(u)
                u*=math.sqrt(l.a)
                l.S_hat.data = u

            elif isinstance(l,Linear_USV) and not l.lr:

                u,s,v_t = torch.linalg.svd(l.weight.data,full_matrices = False)
                mean = float(torch.mean(s))
                l.a = mean
                s = torch.diag(torch.ones(s.shape)*mean)
                l.weight.data = u@s@v_t

            elif isinstance(l,Conv2d_USV) and not l.lr:

                w = rearrange(l.weight.data,'f c i j -> f (c i j)')
                u,s,v_t = torch.linalg.svd(w,full_matrices = False)
                mean = float(torch.mean(s))
                l.a = mean
                s = torch.diag(torch.ones(s.shape)*mean)
                w_cond = u@s@v_t
                l.weight.data = rearrange(w_cond,'f (c i j) -> f c i j',f = l.out_channels,c = l.in_channels,i = l.kernel_size[0],j = l.kernel_size[1])


    
    @torch.no_grad()
    def soft_qr(self):

        for l in self.NN.lr_model:
             
            if hasattr(l,'lr') and l.lr:
                q_pru, r_pru = torch.linalg.qr(l.U.data)
                d = torch.diag(r_pru, 0)
                ph = d.sign()
                q_pru *= ph.expand_as(q_pru)
                l.U.data = q_pru

                q_prv, r_prv = torch.linalg.qr(l.V.data)
                d = torch.diag(r_prv, 0)
                ph = d.sign()
                q_prv *= ph.expand_as(q_prv)
                l.V.data = q_prv

                u,s,v_t = torch.linalg.svd(l.S_hat.data,full_matrices = False)
                mean = torch.mean(s)
                # (m+e)<t(m-e) -> next line
                eps = mean*((self.eps-1)/(self.eps+1))
                for i in range(len(s)):
                    s[i] = torch.clamp(s[i],min = mean-eps,max = mean+eps)
                l.S_hat.data = u@torch.diag(s)@v_t
                setattr(l.S_hat.data,'correction',(torch.abs(s[len(s)-1])/(s[0]))**2)
                setattr(l.U,'correction',(u@u.T)*(torch.abs(s[len(s)-1])/(s[0]))**2)
                setattr(l.V,'correction',(v_t.T@v_t)*(torch.abs(s[len(s)-1])/(s[0]))**2)
