#%%
# imports 
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
import warnings
warnings.filterwarnings("ignore", category=Warning)
torch.set_default_dtype(torch.float32)

# low rank convolution class 

class Conv2d_USV(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1,bias = True,rank = None,
    fixed = False,dtype = None,device = None,load_weights = None,mean = False,approx_orth = False)->None:

        """  
        Initializer for the convolutional low rank layer (filterwise), extention of the classical Pytorch's convolutional layer.
        INPUTS:
        in_channels: number of input channels (Pytorch's standard)
        out_channels: number of output channels (Pytorch's standard)
        kernel_size : kernel_size for the convolutional filter (Pytorch's standard)
        dilation : dilation of the convolution (Pytorch's standard)
        padding : padding of the convolution (Pytorch's standard)
        stride : stride of the filter (Pytorch's standard)
        bias  : flag variable for the bias to be included (Pytorch's standard)
        step : string variable ('K','L' or 'S') for which forward phase to use
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        dtype : Type of the tensors (Pytorch standard, to finish)
        """
            
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2d_USV, self).__init__()

        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size,int) else kernel_size
        self.kernel_size_number = self.kernel_size[0] * self.kernel_size[1]
        self.out_channels = out_channels
        self.dilation = dilation if type(dilation)==tuple else (dilation, dilation)
        self.padding = padding if type(padding) == tuple else(padding, padding)
        self.stride = (stride if type(stride)==tuple else (stride, stride))
        self.in_channels = in_channels
        self.load_weights = load_weights
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.fixed = True
        self.mean = mean
        self.approx_orth = approx_orth
        self.weight = torch.nn.Parameter(torch.empty(tuple([self.out_channels, self.in_channels] +list(self.kernel_size)),**factory_kwargs),requires_grad = True)
        self.lr = True if self.rank!=None else False
        self.rmax = int(min([self.out_channels, self.in_channels*self.kernel_size_number]))
        if not self.fixed or rank == None:
            self.rank = None if rank == None else min([rank,self.rmax])
        else:
            self.rank = min([rank,self.out_channels,self.in_channels*self.kernel_size_number])
        self.dynamic_rank = self.rank

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels,**factory_kwargs))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_channels,**factory_kwargs))

        self.reset_parameters()
    
        # Weights and Bias initialization
        if self.load_weights == None:
            self.reset_parameters()
        else:
            param,b = self.load_weights
            self.bias = torch.nn.Parameter(b)
            self.weight = torch.nn.Parameter(param,requires_grad = True)

        if self.lr:

            n,m = self.out_channels,self.in_channels*self.kernel_size_number

            _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(self.rank))))
            U = torch.randn(n,self.rank)
            V = torch.randn(m,self.rank)
            U,_,_ = torch.linalg.svd(U)
            V,_,_ = torch.linalg.svd(V)
            if mean:
                self.a = self.a = float(torch.mean(s_ordered).detach())
            else:
                self.a = 1.
            self.U = torch.nn.Parameter(U[:,:self.rank].to(device),requires_grad=True)             
            self.S_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device),requires_grad=True)                                       
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
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
         # for testing
        # self.original_weight = Parameter(self.weight.reshape(self.original_shape))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)  


    def forward(self, input):

        """  
        forward phase for the convolutional layer.

        """
        
        batch_size,_,_,_ = input.shape

        if not self.lr:

            return F.conv2d(input = input,weight = self.weight,bias = self.bias,stride = self.stride,
                padding = self.padding,dilation = self.dilation)

        else:

            U_hat,S_hat,V_hat = self.U,self.S_hat,self.V

            # inp_unf = F.unfold(input,self.kernel_size,padding = self.padding,stride = self.stride).to(self.device)
            if input.device.type == "mps":
                unfolded = F.unfold(input.to("cpu"), self.kernel_size, padding=self.padding, stride=self.stride)
                inp_unf = unfolded.to(self.device)
            else:
                inp_unf = F.unfold(input, self.kernel_size, padding=self.padding, stride=self.stride)

            if self.bias is None:
                out_unf = (inp_unf.transpose(1, 2).matmul(V_hat) )
                out_unf = (out_unf.matmul(S_hat.t()))
                out_unf = (out_unf.matmul(U_hat.t()) + self.bias).transpose(1, 2)
            else:
                out_h = int(np.floor(((input.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0])+1))
                out_w = int(np.floor(((input.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1])+1))
                out_unf = (inp_unf.transpose(1, 2).matmul(V_hat) )
                out_unf = (out_unf.matmul(S_hat.t()))
                out_unf = (out_unf.matmul(U_hat.t()) + self.bias).transpose(1, 2)

            return out_unf.view(batch_size, self.out_channels, out_h, out_w)
