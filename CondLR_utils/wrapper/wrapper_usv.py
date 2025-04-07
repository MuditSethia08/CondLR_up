#%%
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from CondLR_utils.layers.conv_usv import Conv2d_USV
from CondLR_utils.layers.linear_usv import Linear_USV
from einops import rearrange
import math
import torch.nn.functional as F


class module_usv(torch.nn.Module):

    def __init__(self,model,**kwargs):
        
        super(module_usv, self).__init__()
        self.lr_model = torch.nn.Sequential()
        self.layer = self.get_layers(model)
        self.name = model.name
        default_rank = []
        dims = []
        for l in self.layer:
            if 'linear' in str(l).lower():
                default_rank.append(min([l.in_features,l.out_features])-1)
                dims.append((l.out_features,l.in_features))
            elif 'conv' in str(l).lower():
                default_rank.append(min([l.out_channels,l.in_channels*l.kernel_size[0]*l.kernel_size[1]])-1)
                dims.append((l.out_channels,l.in_channels*l.kernel_size[0]*l.kernel_size[1]))
        n_lr_layers = len(default_rank)
        self.cr = dict(kwargs)['rank'] if 'rank' in dict(kwargs).keys() else 'NA'
        factory_kwargs = {'rank' : default_rank,'fixed' : [True]*n_lr_layers,'device':'cpu','load_fr_weights' : False,
                            'dims': dims,'baseline':False,'mean':False,'approx_orth':False}
        arguments = {**factory_kwargs ,**dict(kwargs)}
        self.arguments = arguments
        flag = not arguments['baseline']
        k = 0
        for l in self.layer:
            if 'conv' in str(l).lower():
                r =  math.ceil((1.-arguments['rank'][k])*min(l.out_channels,l.in_channels*l.kernel_size[0]*l.kernel_size[1])) if flag else None
                new_layer = Conv2d_USV(l.in_channels,l.out_channels,l.kernel_size,l.dilation,
                                            l.padding,l.stride,l.bias is not None,rank = r,
                                            fixed = arguments['fixed'][k],device = arguments['device'],mean = arguments['mean'],
                                            approx_orth = arguments['approx_orth'])
                if arguments['load_fr_weights']:
                    self.load_weight_layer_(new_layer,(str(l),l),r)
                k+=1
                
            elif 'linear' in str(l).lower():
                r = math.ceil((1.-arguments['rank'][k])*min(l.in_features,l.out_features)) if flag else None
                new_layer = Linear_USV(l.in_features,l.out_features,l.bias is not None,rank = r,
                                            fixed = arguments['fixed'][k],device = arguments['device'],mean = arguments['mean'],
                                            approx_orth = arguments['approx_orth'])
                if arguments['load_fr_weights']:
                    self.load_weight_layer_(new_layer,(str(l),l),r)
                k+=1
            else:
                new_layer = l
                if arguments['load_fr_weights']:
                    self.load_weight_layer_(new_layer,(str(l),l),None)
            self.lr_model.append(new_layer)

            if 'device' in self.arguments.keys():
                self.lr_model.to(device = self.arguments['device'])
        self.last_layer_fr()

    def last_layer_fr(self):
        l = self.lr_model[-1]
        self.lr_model[-1] = Linear_USV(l.in_features,l.out_features,l.bias is not None,rank = min(l.in_features,l.out_features), fixed = True,device = self.arguments['device'],mean = self.arguments['mean'], approx_orth = self.arguments['approx_orth'])
    
    def load_weight_layer_(self,new_layer,old_layer,r):

        name,l = old_layer
        if 'conv' in name.lower() and new_layer.rank!=None:
            f,c,u,v = l.weight.shape
            p = l.weight.reshape((f,c*u*v)).to(l.weight.device)
            U,S,V_t = torch.linalg.svd(p)
            U = U[:,0:r]
            S = torch.diag(S)[0:r,0:r]
            V_t = V_t[0:r,:]
            new_layer.U.data = U
            new_layer.S_hat.data[0:r,0:r] = S
            new_layer.V.data = V_t.T
            if l.bias != None:
                new_layer.bias.data = l.bias
        elif 'linear' in name.lower()  and new_layer.rank!=None:
            p = l.weight
            U,S,V_t = torch.linalg.svd(p)
            U = U[:,0:r]
            S = torch.diag(S)[0:r,0:r]
            V_t = V_t[0:r,:]
            new_layer.U.data = U
            new_layer.S_hat.data[0:r,0:r] = S
            new_layer.V.data = V_t.T
            new_layer.bias.data = l.bias
        else:
            new_layer.load_state_dict(l.state_dict())

         
    def get_layers(self,model: torch.nn.Module):

        children = list(model.children())
        flatt_children = []
        if children == []:

            return model
        else:

            for child in children:
                    try:
                        flatt_children.extend(self.get_layers(child))
                    except TypeError:
                        flatt_children.append(self.get_layers(child))
            return flatt_children
        
    @torch.no_grad()
    def get_layers_cond(self):
        conds = []
        for l in self.lr_model:
            if hasattr(l,'lr') and l.lr:
                conds.append(torch.linalg.cond(l.S_hat))
            else:
                if isinstance(l,torch.nn.Conv2d):
                    conds.append(torch.linalg.cond(rearrange(l.weight,'f c u v -> f (c u v)')))
                elif isinstance(l,torch.nn.Linear):
                    conds.append(torch.linalg.cond(l.weight))
        return conds

    def forward_all(self,x):
        return self.lr_model(x)

    def forward_block(self, x, n_c, i_inner):
        if i_inner == 0:
            x = self.lr_model[n_c: n_c + 2](x)
            out = self.lr_model[n_c + 2](x)
        else:
            out = self.lr_model[n_c: n_c + 3](x)
        out = self.lr_model[n_c + 3:n_c + 6](out)
        if i_inner == 0:
            torch.add(self.lr_model[n_c + 6](x), out)
        else:
            torch.add(x, out)
        return out
    def forward_wrn(self, x):
        out = self.lr_model[0](x)
        n_c = 1
        add_arr = [7, 6, 6, 6]
        for i_outer in range(3):
            for i_inner in range(2):#(4):
                out = self.forward_block(out, n_c, i_inner)
                n_c += add_arr[i_inner]
        out = self.lr_model[n_c : n_c + 2](out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.lr_model[n_c + 2].in_features)
        out = out = self.lr_model[n_c + 2](out)
        return out
    
    def forward(self,x):
        if self.name == 'wrn':
            return self.forward_wrn(x)
        else:
            return self.forward_all(x)

    def update_device(self,device):
        for l in self.lr_model:
            if hasattr(l,'device'):
                l.device = device

    def eval(self):

        return self.train(False)
    
    @torch.no_grad()
    def get_ranks(self):
        ranks = []
        for l in self.lr_model:
            if hasattr(l,'lr') and l.lr:
                ranks.append(l.rank)
            elif isinstance(l,Conv2d_USV) and not l.lr:
                ranks.append(min(l.out_channels,l.in_channels*l.kernel_size[0]*l.kernel_size[1]))
            elif isinstance(l,Linear_USV) and not l.lr:
                ranks.append(min(l.in_features,l.out_features))
        return ranks
    
    @torch.no_grad()
    def get_conds(self):
        condition_numbers = []
        for l in self.lr_model:
            if hasattr(l,'lr') and l.lr:
                c = float(torch.linalg.cond(l.S_hat))
                condition_numbers.append(c)
            elif isinstance(l,Linear_USV) and not l.lr:
                w  = l.weight.T
                c = float(torch.linalg.cond(w))
                condition_numbers.append(c)
            elif isinstance(l,Conv2d_USV) and not l.lr:
                w = rearrange(l.weight,'f c i j -> f (c i j)')
                c = float(torch.linalg.cond(w))
                condition_numbers.append(c)
        return condition_numbers
    
    @torch.no_grad()
    def get_mean_svs(self):
        mean_svs = []
        for l in self.lr_model:
            if hasattr(l,'lr') and l.lr:
                # c = l.a
                m = torch.diag(l.S_hat@l.S_hat.T)
                c = float(torch.mean(m))
                sd = float(torch.std(m))
                mean_svs.append((c,sd))
            elif isinstance(l,Linear_USV) and not l.lr:
                w  = l.weight.T
                m = torch.diag(w@w.T)
                c = float(torch.mean(m))
                sd = float(torch.std(m))
                mean_svs.append((c,sd))
            elif isinstance(l,Conv2d_USV) and not l.lr:
                w = rearrange(l.weight,'f c i j -> f (c i j)')
                m = torch.diag(w@w.T)
                c =  float(torch.mean(m))
                sd = float(torch.std(m))
                mean_svs.append((c,sd))
        return mean_svs


    def populate_gradients(self,x,y,criterion):

        loss = criterion(self.lr_model(x),y)
        loss.backward()
        return loss
