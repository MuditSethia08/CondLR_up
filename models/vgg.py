import os 
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import numpy as np

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

class Flatten(torch.nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        # out = input.view(batch_size,-1)
        out = input.contiguous().view(batch_size,-1)
        return out 



class VGG(torch.nn.Module):
    def __init__(
        self,
        architecture,
        in_channels=3, 
        in_height=224, 
        in_width=224, 
        num_hidden=4096,
        num_classes=1000,
        bn = True
    ):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.name = 'vgg'
        self.layer = torch.nn.Sequential()
        j = 0
        for x in architecture:
            if type(x) == int:
                out_channels = x
                if bn:
                    bias_flag=False
                else:
                    bias_flag=True
                self.layer.add_module('conv_'+str(j),torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=(1,1),
                            padding=(1, 1),
                            bias=bias_flag
                        ))
                if bn:
                    self.layer.add_module('bn_'+str(j),torch.nn.BatchNorm2d(out_channels,momentum = 0.9))
                self.layer.add_module('relu_'+str(j),torch.nn.LeakyReLU(inplace=True))
                in_channels = x
            else:
                self.layer.add_module('maxpool_'+str(j),
                    torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )
            j+=1

        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(f"`in_height` and `in_width` must be multiples of {factor}")
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )

        self.layer.add_module('flat',Flatten())
        self.layer.add_module('linear_'+str(1),torch.nn.Linear(
                last_out_channels * out_height * out_width, 
                self.num_hidden))
        self.layer.add_module('relu_'+str(j+1),torch.nn.LeakyReLU(inplace=True))
        self.layer.add_module('linear_'+str(2),torch.nn.Linear(self.num_hidden, self.num_hidden,
                            ))
        self.layer.add_module('relu_'+str(j+2),torch.nn.LeakyReLU(inplace=True))
        self.layer.add_module('drop_2',torch.nn.Dropout(p=0.2))
        self.layer.add_module('classifier',torch.nn.Linear(self.num_hidden, self.num_classes))
        self.init_weights()

    def init_weights(self,name  = 'kn'):
        if name == 'kn':
            for l in self.layer:
                if isinstance(l, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(l.weight,nonlinearity='relu')
                    torch.nn.init.uniform(l.bias.data)
                elif isinstance(l,torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(l.weight,nonlinearity='relu')
                    if l.bias != None:
                        torch.nn.init.uniform(l.bias.data)
        elif name == 'orthogonal':
            for l in self.layer:
                if isinstance(l,torch.nn.Linear) or isinstance(l,torch.nn.Conv2d):
                    torch.nn.init.orthogonal(l.weight.data,gain=1.41)
                    if l.bias != None:
                        torch.nn.init.constant(l.bias.data,val = 0.0)
        
    def forward(self, x):
        return self.layer(x)
