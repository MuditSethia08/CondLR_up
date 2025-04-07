import math       
import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)

class Linear_USV(torch.nn.Module):
    

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,rank = None,fixed = False,load_weights = None,mean = False,approx_orth = False) -> None:

        """  
        initializer for the low rank linear layer, extention of the classical Pytorch's Linear
        INPUTS:
        in_features : number of inputs features (Pytorch standard)
        out_features : number of output features (Pytorch standard)
        bias : flag for the presence of bias (Pytorch standard)
        device : device in where to put parameters
        dtype : type of the tensors (Pytorch standard)
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_USV, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.rank = rank
        self.device = device
        self.load_weights = load_weights
        self.fixed = True #fixed
        self.approx_orth = approx_orth
        self.lr = True if self.rank!=None else False
        self.rmax = int(min([self.in_features, self.out_features]) / 2)
        if not self.fixed or rank==None:
            self.rank = None if rank == None else min([rank,self.rmax])
        else:
            self.rank = min([rank,self.in_features,self.out_features])
        self.dynamic_rank = self.rank

        if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
                self.register_parameter('bias', None)

        self.reset_parameters()

        self.mean = True
        if self.lr:

            U,S,V = torch.linalg.svd(self.weight)
            if mean:
                self.a = float(torch.mean(S).detach())
            else:
                self.a = 1.
            S = torch.diag(S[:self.rank])
            U = U[:,:self.rank]
            V = V.T[:,:self.rank]
            self.U = torch.nn.Parameter(U[:,:self.rank].to(device) ,requires_grad=True)           
            self.S_hat = torch.nn.Parameter(S.to(device),requires_grad=True)                                   
            self.V = torch.nn.Parameter(V[:,:self.rank].to(device),requires_grad=True)
            setattr(self.U,'U',1)
            setattr(self.V,'V',1)
            setattr(self.S_hat,'S_hat',1)
            ###################
            if self.mean:
                setattr(self.S_hat,'a',self.a)
                setattr(self.U,'a',self.a)
                setattr(self.V,'a',self.a)
            ###################
            if self.approx_orth:
                setattr(self.U,'correction',torch.eye(self.U.shape[1]).to(self.U.device))
                setattr(self.V,'correction',torch.eye(self.V.shape[1]).to(self.V.device))
                setattr(self.S_hat,'correction',1)
                setattr(self.S_hat,'mean',self.a)
                setattr(self.U,'mean',self.a)
                setattr(self.V,'mean',self.a)
            self.weight = None
        else:
            self.a = 1
        


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """  
        forward phase for the layer
        """
        if not self.lr:

            x = F.linear(input,self.weight,self.bias)

        else : 

                S_hat,U_hat,V_hat = self.S_hat[:self.rank,:self.rank],self.U[:,:self.rank],self.V[:,:self.rank]

                x = F.linear(input,V_hat.T)
                x = F.linear(x,S_hat)
                x = F.linear(x,U_hat)
                if self.bias is not None:
                    x = x+self.bias            
        return x 