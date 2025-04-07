import torch 

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
