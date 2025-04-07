# Pytorch repository for "Robust low-rank training via approximate orthonormal constraints"
## Paper published at NeurIPS 2023


### Installation

1. Create a conda environment 
2. Install pip inside the new conda environment (``conda install pip``)
3. Install the project requirements inside the conda environment with ``pip install -r requirements.txt``

### test cases
The folder "scripts" contains the test cases.
To run one of the bash scripts, from the current path execute ``sh scripts/test_case_name.sh``.

### Creation of a low-rank network to optimize

In order to use CondLR it is necessary to create a custom torch.nn.Module to feed into the optimizer.
It can be done as explained in the following steps:

1. Create a standard torch.nn.Module with the network you need;
2. Use the module_usv in the wrapper folder with the parameters of your choice to wrap the previous network;
3. Now you have an instance that can be optimized using the custom CondLR optimizer.

### Training 

Now the training of this new low-rank network can be performed using the custom CondLR optimizer.


## Example use
```
import torch
from wrapper.wrapper_usv import module_usv

# EXAMPLE OF PYTORCH MODULE TO WRAP
class Lenet5(torch.nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.name = 'lenet5'
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride=1),  
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5, stride=1),  
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(800,out_features = 500),  
            torch.nn.LeakyReLU(),
            torch.nn.Linear(500,out_features = 10)
        )

    def forward(self, x):
        return self.layer(x)


# CREATION OF LOW_RANK NETWORK
cr = 0.8
mean = True
approx_orth = not mean
NN = module_usv(f,rank = [cr,cr,cr]+[0.0],device = args.device,
                       baseline = baseline,mean = mean,approx_orth = approx_orth)

# INTIALIZATION OF THE OPTIMIZER
optimizer = opt_USV(f,**kwargs)

criterion = torch.nn.CrossEntropyLoss() 
trans = transforms.Compose(transforms.ToTensor())
train_loader = datasets.MNIST(root='./data', train=True, download=True, transform=trans)

# TRAIN LOOP
NN.train()
for i,data in enumerate(train_loader):  # train
      for param in NN.parameters():
         param.grad = None
      inputs,labels = data
      inputs,labels = inputs.to(device),labels.to(device)
      outputs = NN(inputs).to(device)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.usv_step()

# EVALUATION 
NN.eval()
evaluation_function()   # your custom evaluation function
```

### LOAD AND SAVE MODELS
### Example use

```
# TO SAVE AND LOAD A LOW-RANK MODEL

torch.save(f,'path_save.pt')
f = torch.load('path_save.pt')

```


