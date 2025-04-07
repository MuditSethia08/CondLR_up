import os 
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import argparse
from CondLR_utils.trainer_USV import train_and_regularize_FGSM
from CondLR_utils.optimizer_LR.usv_optimizer import opt_USV
from models.lenet5 import Lenet5
from models.vgg import VGG,VGG_types
from models.wrn import WideResNet
from CondLR_utils.wrapper.wrapper_usv import module_usv

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

############################################## parser creation
parser = argparse.ArgumentParser(description='Pytorch CondLR training ')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--load_w', type=bool, default=True, metavar='LOAD_W',
                    help='Load weights')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate for optimizer (default: 0.05)') 
parser.add_argument('--wd', type=float, default=0.0, metavar='WD',
                    help='regularization for Frobenius norm (default: 0.0)') 
parser.add_argument('--beta', type=float, default=0.0, metavar='BETA',
                    help='regularization for PARSEVAL (default: 0.0)') 
parser.add_argument('--cr', type=float, default=-1, metavar='cr',   #[0.8]
                     help='compression ratio') 
parser.add_argument('--scheduler', type=str, default='plateau', metavar='scheduler',
                     help='scheduler') 
parser.add_argument('--momentum', type=float, default=0.0, metavar='MOMENTUM',
                    help='momentum (default: 0.1)')  
parser.add_argument('--workers', type=int, default=1, metavar='WORKERS',
                    help='number of workers for the dataloaders (default: 1)')   
parser.add_argument('--cv_run', type=int, default=1, metavar='CV_RUN',
                    help='number of runs (default: 1)')
parser.add_argument('--stiefel_opt', type=str, default='mean', metavar='stiefel option',
                    help='method option: mean, approx_orth, cayley_sgd') 
parser.add_argument('--retraction_opt', type=str, default='qr', metavar='stiefel option',
                    help='retraction option for cayley_sgd: qr, cayley, cayley_full') 
parser.add_argument('--eps', type=float, default=0.0, metavar='EPS',
                    help='tolerance for maximal layer condition number in the (default:0.1, working only with approx_orth True)')        
parser.add_argument('--save_weights', type=bool, default=True, metavar='SAVE_WEIGHTS',
                    help='save the weights of the best validation model during the run (default: True)') 
parser.add_argument('--save_progress', type=bool, default=True, metavar='SAVE_PROGRESS',
                    help='save running data during the training (default: False)') 
parser.add_argument('--device', type=str, default='cuda', metavar='DEVICE',
                    help='device to use (default: CUDA)') 
parser.add_argument('--run_name', type=str, default='', metavar='RUNNAME',
                    help='name to save the run (default one present)') 
parser.add_argument('--p_budget', type=float, nargs='+', default=[0.], metavar='perturbation budget',
                     help='list of compression ratios (default: [0.0,0.1,0.2,...,1.0])')
parser.add_argument(
    "--attack_name",
    default='fgsm', 
    choices=["fgsm","pgd_inf","pgd_l2"])
parser.add_argument(
    "--net_name",
    default='lenet5', 
    choices=["lenet5","vgg16","wrn"])
parser.add_argument(
    "--dataset_name",
    default='mnist', 
    choices=["mnist", "cifar10", "cifar100", "svhn"])
args = parser.parse_args()
############################################## Net creation
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
args.device = device if args.device == 'cuda' else 'cpu'
print(f"Using {device} device")

MAX_EPOCHS = args.epochs

def accuracy(outputs,labels):

    return torch.sum(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = torch.float16))


metric  = accuracy
criterion = torch.nn.CrossEntropyLoss() 
metric_name = 'accuracy'
dataset_path = './data'

##### WTW or WWT, depends on the dims of the matrix, scaricare fattori della norma delle basi su beta
for cvr_i in range(args.cv_run):
    if args.dataset_name == 'cifar10':

        trans = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trans_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=trans)
        val_loader  = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=trans_test)

        val_prop=0.25
        tr_size = len(train_loader)
        indices = np.arange(len(train_loader))
        np.random.shuffle(indices)
        split=int(np.floor(val_prop*tr_size))
        train_idx=indices[split:]
        valid_idx=indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(valid_idx)

        test_loader = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=trans_test)


        train_loader = DataLoader(
            train_loader,
            batch_size=args.batch_size, sampler=train_sampler, num_workers = args.workers)

        val_loader = DataLoader(
            val_loader,
            batch_size=args.batch_size, sampler=val_sampler,num_workers = args.workers)

        test_loader = DataLoader(
            test_loader,
            batch_size=64, shuffle=False,num_workers = args.workers)
    elif args.dataset_name == 'cifar100':
        trans = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        trans_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        train_loader = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=trans)
        val_loader  = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=trans_test)
        val_prop=0.25
        tr_size = len(train_loader)
        indices = np.arange(len(train_loader))
        np.random.shuffle(indices)
        split=int(np.floor(val_prop*tr_size))
        train_idx=indices[split:]
        valid_idx=indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(valid_idx)
        test_loader = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=trans_test)

        train_loader = DataLoader(
                train_loader,
                batch_size=args.batch_size, sampler=train_sampler, num_workers = args.workers)
        val_loader = DataLoader(
                val_loader,
                batch_size=args.batch_size, sampler=val_sampler,num_workers = args.workers)
        test_loader = DataLoader(
                test_loader,
                batch_size=64, shuffle=False,num_workers = args.workers)
    elif args.dataset_name == 'mnist':

        trans = T.Compose([T.ToTensor()])
        train_loader = datasets.MNIST(root=dataset_path, train=True, download=True, transform=trans)
        val_loader =  datasets.MNIST(root=dataset_path, train=True, download=True, transform=trans)

        val_prop=0.25
        tr_size = len(train_loader)
        indices = np.arange(len(train_loader))
        np.random.shuffle(indices)
        split=int(np.floor(val_prop*tr_size))
        train_idx=indices[split:]
        valid_idx=indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(valid_idx)

        test_loader = datasets.MNIST(root=dataset_path, train=False, download=True, transform=trans)

        train_loader = DataLoader(
            train_loader,
            batch_size=args.batch_size, sampler=train_sampler, num_workers = args.workers)
    
        val_loader = DataLoader(
            val_loader,
            batch_size=args.batch_size, sampler=val_sampler, num_workers = args.workers)

        test_loader = DataLoader(
            test_loader,
            batch_size=64, shuffle=False,num_workers = args.workers)
    elif args.dataset_name == 'svhn':
        trans = T.Compose([T.ToTensor()])
        train_loader = datasets.SVHN(root=dataset_path, split='train', download=True, transform=trans)
        val_loader =  datasets.SVHN(root=dataset_path, split='train', download=True, transform=trans)
    
        val_prop=0.25
        tr_size = len(train_loader)
        indices = np.arange(len(train_loader))
        np.random.shuffle(indices)
        split=int(np.floor(val_prop*tr_size))
        train_idx=indices[split:]
        valid_idx=indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        val_sampler=SubsetRandomSampler(valid_idx)

        test_loader = datasets.SVHN(root=dataset_path, split='test', download=True, transform=trans)
    
        train_loader = DataLoader(train_loader,batch_size=args.batch_size, sampler=train_sampler, num_workers = args.workers)
        val_loader = DataLoader(val_loader,batch_size=args.batch_size, sampler=val_sampler, num_workers = args.workers)
        test_loader = DataLoader(test_loader,batch_size=64, shuffle=False,num_workers = args.workers)

##### WTW or WWT, depends on the dims of the matrix, scaricare fattori della norma delle basi su beta
#for cvr_i in range(args.cv_run):
    cr = args.cr
    if cr == -1:
        cr='False'

    baseline = (cr=='False')

    if args.net_name == 'lenet5':
        f = Lenet5()
        f = module_usv(f,rank = [args.cr,args.cr,args.cr]+[0.0],device = args.device,
                        baseline = baseline,mean = args.stiefel_opt== 'mean',approx_orth = args.stiefel_opt== 'approx_orth')
        f.to(args.device)

    elif args.net_name == 'vgg16':
        f = VGG(VGG_types['VGG16'],3,32,32,512,10)
        f = module_usv(f,rank = [args.cr]*(len(f.layer)-1)+[0.0],device = args.device,
                        baseline = baseline,mean = args.stiefel_opt== 'mean',approx_orth = args.stiefel_opt== 'approx_orth')
        f.to(args.device)
    elif args.net_name == 'wrn':
        if args.dataset_name == 'cifar100':
            num_cl = 100
        else:
            num_cl = 10
        f = WideResNet(16, num_cl, widen_factor=4)
        f = module_usv(f,rank = [args.cr]*(17-1)+[0.0],device = args.device,
                baseline = baseline,mean = args.stiefel_opt== 'mean',approx_orth = args.stiefel_opt== 'approx_orth')
        f.to(args.device)


    print(f'train baseline : {baseline}, cr: {cr}, lr, mom, wd {args.lr, args.momentum,  args.wd}, approx_orth {args.stiefel_opt}, mean collapse {args.stiefel_opt}')
    print('='*40)


    optimizer = opt_USV(f,baseline = baseline,lr = args.lr,momentum = args.momentum, weight_decay = args.wd, stiefel_opt = args.stiefel_opt, retraction_opt = args.retraction_opt,eps = args.eps)
    if args.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer.integrator,gamma = 0.998)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.integrator,factor=0.4)
    elif args.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.integrator, milestones=[70, 100], gamma=0.3)


    path = './results_USV/'
    save_name = f'{args.net_name}_{args.dataset_name}_mom{args.momentum}_lr{args.lr}_wd{args.wd}_cr{cr}_baseline{baseline}_cv{cvr_i}' if args.run_name == '' else args.run_name + f'_cv{cvr_i}'

    #### rechange to train  ###########
    if __name__ == "__main__":
    # your script's current logic goes here

        train_and_regularize_FGSM(f,optimizer=optimizer,criterion = criterion,train_loader=train_loader,
                                    validation_loader = val_loader,test_loader=test_loader,
                                    metric = accuracy,epochs = args.epochs,device = args.device,
                                    path = path,save_weights=args.save_weights,
                                    save_progress=args.save_progress,scheduler=scheduler,save_name =save_name,args = args)


