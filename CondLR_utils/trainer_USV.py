import pandas as pd
import torch
from torch import float16
import numpy as np
import time
from tqdm import tqdm

def pgd_l2_attack(model, loss_f, image, target, eps=0.006, alpha=0.002, dataset='mnist', random_start=True, steps=10, eps_for_division=1e-10):
    """
    modified from the library: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/index.html

    Distance Measure : L2

    Arguments:
        model: model to attack.
        eps: maximum perturbation.
        alpha: step size.
        steps: number of steps.
        random_start: using random initialization of delta. 

        loss_f: loss function 
        image, target: picture-label pair
        dataset: name of the dataset
    """
    model.eval()
    if dataset == 'mnist':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'fashion_mnist':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'cifar10' or dataset == 'cifar100':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        means = torch.tensor([0.4914, 0.4822, 0.4465])
    image = image.clone().detach()
    target = target.clone().detach()
    if dataset == 'cifar10' or dataset == 'cifar100':
        image[:, 0, :, :] = image[:, 0, :, :] * std[0] + means[0]
        image[:, 1, :, :] = image[:, 1, :, :] * std[1] + means[1]
        image[:, 2, :, :] = image[:, 2, :, :] * std[2] + means[2]
    adv_image = image.clone().detach()
    batch_size = len(image)
    if random_start:
        # Starting at a uniformly random point
        delta = torch.empty_like(adv_image).normal_()
        d_flat = delta.view(adv_image.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(adv_image.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*eps
        adv_image = torch.clamp(adv_image + delta, min=0, max=1).detach()
    for _ in range(steps):
        adv_image_norm = adv_image.clone().detach()
        if dataset == 'cifar10' or dataset == 'cifar100':
            adv_image_norm[:, 0, :, :] = (adv_image_norm[:, 0, :, :] - means[0]) / std[0]
            adv_image_norm[:, 1, :, :] = (adv_image_norm[:, 1, :, :] - means[1]) / std[1]
            adv_image_norm[:, 2, :, :] = (adv_image_norm[:, 2, :, :] - means[2]) / std[2]
        adv_image_norm.requires_grad = True
        output = model(adv_image_norm)
        loss = loss_f(output, target)
        # Update adversarial images
        grad = torch.autograd.grad(loss, adv_image_norm, retain_graph=False, create_graph=False)[0]
        grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + eps_for_division  # nopep8
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        adv_image = adv_image.detach() + alpha * grad
        delta = adv_image - image
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = eps / (delta_norms+ eps_for_division)
        factor = torch.min(factor, torch.ones_like(delta_norms))
        #print(factor)
        delta = delta * factor.view(-1, 1, 1, 1)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    adv_image_norm = adv_image.clone().detach()
    if dataset == 'cifar10' or dataset == 'cifar100':
        adv_image_norm[:, 0, :, :] = (adv_image_norm[:, 0, :, :] - means[0]) / std[0]
        adv_image_norm[:, 1, :, :] = (adv_image_norm[:, 1, :, :] - means[1]) / std[1]
        adv_image_norm[:, 2, :, :] = (adv_image_norm[:, 2, :, :] - means[2]) / std[2]
    return adv_image_norm



def pgd_attack(model, loss_f, image, target, eps=0.006, alpha=0.002, dataset='mnist', random_start=True, steps=10):
    """
    modified from the library: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/index.html

    Distance Measure : Linf

    Arguments:
        model: model to attack.
        eps: maximum perturbation.
        alpha: step size.
        steps: number of steps.
        random_start: using random initialization of delta. 

        loss_f: loss function 
        image, target: picture-label pair
        dataset: name of the dataset
    """
    if dataset == 'mnist':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'fashion_mnist':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'cifar10' or dataset == 'cifar100':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        means = torch.tensor([0.4914, 0.4822, 0.4465])
    image = image.clone().detach()
    target = target.clone().detach()
    if dataset == 'cifar10' or dataset == 'cifar100':
        image[:, 0, :, :] = image[:, 0, :, :] * std[0] + means[0]
        image[:, 1, :, :] = image[:, 1, :, :] * std[1] + means[1]
        image[:, 2, :, :] = image[:, 2, :, :] * std[2] + means[2]
    adv_image = image.clone().detach()                                                                                                                  
    if random_start:
        adv_image = adv_image + torch.empty_like(adv_image).uniform_(-eps, eps)
        adv_image = torch.clamp(adv_image, clip_min, clip_max).detach()
    for _ in range(steps):
        adv_image_norm = adv_image.clone().detach()
        if dataset == 'cifar10' or dataset == 'cifar100':
            adv_image_norm[:, 0, :, :] = (adv_image_norm[:, 0, :, :] - means[0]) / std[0]
            adv_image_norm[:, 1, :, :] = (adv_image_norm[:, 1, :, :] - means[1]) / std[1]
            adv_image_norm[:, 2, :, :] = (adv_image_norm[:, 2, :, :] - means[2]) / std[2]
        adv_image_norm.requires_grad = True
        output = model(adv_image_norm)
        loss = loss_f(output, target)
        # Update adversarial images
        grad = torch.autograd.grad(loss, adv_image_norm, retain_graph=False, create_graph=False)[0]
        adv_image = adv_image.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    adv_image_norm = adv_image.clone().detach()
    if dataset == 'cifar10' or dataset == 'cifar100':
        adv_image_norm[:, 0, :, :] = (adv_image_norm[:, 0, :, :] - means[0]) / std[0]
        adv_image_norm[:, 1, :, :] = (adv_image_norm[:, 1, :, :] - means[1]) / std[1]
        adv_image_norm[:, 2, :, :] = (adv_image_norm[:, 2, :, :] - means[2]) / std[2]
    return adv_image_norm

def fgsm_attack(model, loss_f, image, target, epsilons, dataset='mnist'):
    """
    modified from the library: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/index.html

    Distance Measure : Linf

    Arguments:
        model: model to attack.
        epsilons: perturbation size.

        loss_f: loss function 
        image, target: picture-label pair
        dataset: name of the dataset
    """
    if dataset == 'mnist':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'fashion_mnist':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'svhn':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([1.])
    elif dataset == 'cifar10' or dataset == 'cifar100':
        clip_min = 0.
        clip_max = 1.
        std = torch.tensor([0.2023, 0.1994, 0.2010])
    image = image.clone().detach()
    target = target.clone().detach()
    image.requires_grad = True
    output = model(image)
    loss = loss_f(output, target)
    data_grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
    sign_data_grad = data_grad.sign()
    perturbed_images = []
    if dataset == 'cifar10' or dataset == 'cifar100':
        sign_data_grad[:, 0, :, :] = sign_data_grad[:, 0, :, :]/std[0]
        sign_data_grad[:, 1, :, :] = sign_data_grad[:, 1, :, :]/std[1]
        sign_data_grad[:, 2, :, :] = sign_data_grad[:, 2, :, :]/std[2]
    for eps in epsilons:
        perturbed_img = image + eps*sign_data_grad
        if dataset != 'cifar10' and dataset != 'cifar100':
            perturbed_img = torch.clamp(perturbed_img, clip_min, clip_max).detach()
        perturbed_images.append(perturbed_img.detach())
    return perturbed_images

def attack_choice(NN, criterion, inputs, labels,attack_name="fgsm", budget=[0., 0.01], dataset='mnist'):
    if attack_name == 'fgsm':
        perturbed_data = fgsm_attack(NN, criterion, inputs, labels, epsilons=budget,dataset=dataset)
    elif attack_name == 'pgd_inf':
        pgd_steps = 10
        for pgd_eps in budget:
            pgd_data = pgd_attack(NN, criterion, inputs, labels, eps=pgd_eps, alpha=1.25*pgd_eps/pgd_steps, steps=pgd_steps, dataset=dataset)
            perturbed_data.append(pgd_data)
    elif attack_name == 'pgd_l2':
        pgd_steps = 10
        for pgd_eps in budget:
            pgd_data = pgd_l2_attack(NN, criterion, inputs, labels, eps=pgd_eps, alpha=1.25*pgd_eps/pgd_steps, steps=pgd_steps, dataset=dataset)
            perturbed_data.append(pgd_data)
    return perturbed_data


def train_and_regularize_FGSM(NN,optimizer,train_loader,validation_loader,test_loader,criterion,metric,epochs,
                metric_name = 'accuracy',device = 'cpu',path = '',
                scheduler = None,save_weights = True,save_progress = False,save_name = '',args = None):

    """ 
    INPUTS:
    NN : neural network with custom layers and methods 
    train/validation/test_loader : loader for datasets
    criterion : loss function
    metric : metric function
    epochs : number of epochs to train
    metric_name : name of the used metric
    path : path string for where to save the results
    OUTPUTS:
    running_data : Pandas dataframe with the results of the run
    """

    running_data = pd.DataFrame(data = None,columns = ['epoch','cr','learning_rate','train_loss','train_'+metric_name+'(%)','validation_loss',\
                                                        'validation_'+metric_name+'(%)','test_'+metric_name+'(%)',\
                                                     'fgsm_list', 'ranks','conds','mean_svs','timing batch forward'])

    # beta = optimizer.beta
    def accuracy(outputs,labels):

        return torch.sum(torch.argmax(outputs.detach(),axis = 1) == labels).clone().detach()

    metric = accuracy
    batch_size = train_loader.batch_size

    fgsm_epoch_accs = []
    ep_fgsm = 20

    for epoch in tqdm(range(epochs)):

        print(f'epoch {epoch}---------------------------------------------')
        loss_hist = 0
        acc_hist = 0
        batch_size = train_loader.batch_size
        k = len(train_loader)
        average_batch_time = 0.0

        NN.train()
        for i,data in enumerate(train_loader):  # train
            for param in NN.parameters():
                param.grad = None
            start = time.time()
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = NN(inputs)#.to(device)
            loss = criterion(outputs,labels)
            loss.backward()
            loss_hist+=float(loss.item())/(k*batch_size)
            acc_hist += float(metric(outputs.detach(),labels))/(k*batch_size)
            optimizer.usv_step()
            stop = time.time() - start
            average_batch_time+=stop/k

        NN.eval()
        with torch.no_grad():
            k = len(validation_loader)
            loss_hist_val = 0.0
            acc_hist_val = 0.0
            batch_size = validation_loader.batch_size
            for i,data in enumerate(validation_loader):   # validation 
                inputs,labels = data
                inputs,labels = inputs.to(device),labels.to(device)
                outputs = NN(inputs).detach().to(device)
                loss_val = criterion(outputs,labels)
                loss_hist_val+=float(loss_val.item())/(k*batch_size)
                acc_hist_val += float(metric(outputs,labels))/(k*batch_size)


        if test_loader!=None and (epoch%ep_fgsm == 0 or epoch == epochs-1) and epoch>0:
            k = len(test_loader)
            loss_hist_test = 0.0
            acc_hist_test= 0.0
            batch_size = test_loader.batch_size
            accs_fgsm = np.zeros(len(args.p_budget))
            NN.eval()
            for i,data in enumerate(test_loader):   # validation 
                inputs,labels = data
                inputs,labels = inputs.to(device),labels.to(device)
                outputs = NN(inputs).detach().to(device)
                loss_test = criterion(outputs,labels)
                perturbed_data = attack_choice(NN, criterion, inputs, labels, attack_name=args.attack_name, budget=args.p_budget, dataset=args.dataset_name)
                for pim_i, pert_img in enumerate(perturbed_data):
                    output_adv = NN(pert_img).detach().to(device)
                    accs_fgsm[pim_i] += float(metric(output_adv,labels))/(k*batch_size)
            
                loss_hist_test += float(loss_test.item())/(k*batch_size)
                acc_hist_test += float(metric(outputs,labels))/(k*batch_size)

            final_acc = accs_fgsm
            fgsm_epoch_accs.append(list(final_acc))
            print(fgsm_epoch_accs)

        else:
            final_acc = [-1]
            loss_hist_test = -1
            acc_hist_test = -1

        print(f'epoch[{epoch}]: loss: {loss_hist:9.8f} | {metric_name}: {acc_hist:9.4f} | val loss: {loss_hist_val:9.8f} | val {metric_name}:{acc_hist_val:9.4f}')
        print('='*100)
        ranks = NN.get_ranks()
        conds = NN.get_conds()
        mean_svs = NN.get_mean_svs()
        with torch.no_grad():
            i_r = 0
            for r,c,m in zip(ranks,conds,mean_svs):
                print(f'rank layer {i_r}: {r}, cond: {c}, mean s.v. (and std): {m}')
                i_r += 1
            print('\n')


        epoch_data = [epoch,NN.cr,round(optimizer.integrator.param_groups[0]["lr"],5),round(loss_hist,8),round(acc_hist*100,4),round(loss_hist_val,8),\
                    round(acc_hist_val*100,4),round(acc_hist_test*100,4), final_acc,ranks,conds,mean_svs,average_batch_time]

        running_data.loc[epoch] = epoch_data

        if save_name is not None and (epoch%5 == 0 or epoch == epochs-1) and save_progress:

            running_data.to_csv(path+save_name+'.csv')

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(loss_hist)
            else:
                scheduler.step()

        if epoch == 0:

            best_val_loss = loss_hist_val
            best_val_acc = acc_hist_val

        if loss_hist_val<best_val_loss and save_weights:
        #    print('save best loss', epoch)
        #    torch.save(NN,path+save_name+'_last_best_val.pt')
            best_val_loss = loss_hist_val
        if acc_hist_val>=best_val_acc and save_weights:
            print('save best acc', epoch)
            torch.save(NN,path+save_name+'_best_acc.pt')
            best_val_acc = acc_hist_val
    torch.save(NN,path+save_name+'_last_model.pt')

    model_best = torch.load(path+save_name+'_last_model.pt', map_location=device) 
    model_best.eval()
    pgd_steps = 10
    if test_loader!=None:
        k = len(test_loader)
        loss_hist_test = 0.0
        acc_hist_test= 0.0
        batch_size = test_loader.batch_size
        accs_fgsm = np.zeros(len(args.p_budget))
        for i,data in enumerate(test_loader):   # validation 
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model_best(inputs).detach().to(device)
            loss_test = criterion(outputs,labels)
            perturbed_data = attack_choice(NN, criterion, inputs, labels, attack_name=args.attack_name, budget=args.p_budget, dataset=args.dataset_name)
            for pim_i, pert_img in enumerate(perturbed_data):
                output_adv = model_best(pert_img).detach().to(device)
                accs_fgsm[pim_i] += float(metric(output_adv,labels))/(k*batch_size)
        
            loss_hist_test += float(loss_test.item())/(k*batch_size)
            acc_hist_test += float(metric(outputs,labels))/(k*batch_size)

        final_acc = accs_fgsm
        print('last model robustness', final_acc)
        epoch_data = [epochs,model_best.cr,round(optimizer.integrator.param_groups[0]["lr"],5),-1,-1,-1,\
                        -1,round(acc_hist_test*100,4), final_acc,model_best.get_ranks(),model_best.get_conds(),model_best.get_mean_svs(),-1]

        running_data.loc[epochs] = epoch_data

        if save_name is not None and save_progress:
            running_data.to_csv(path+save_name+'.csv')
    model_best = torch.load(path+save_name+'_best_acc.pt', map_location=device)
    model_best.eval()
    pgd_steps = 10
    if test_loader!=None:
        k = len(test_loader)
        loss_hist_test = 0.0
        acc_hist_test= 0.0
        batch_size = test_loader.batch_size
        accs_fgsm = np.zeros(len(args.p_budget))
        for i,data in enumerate(test_loader):   # validation
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model_best(inputs).detach().to(device)
            loss_test = criterion(outputs,labels)
            perturbed_data = attack_choice(NN, criterion, inputs, labels, attack_name=args.attack_name, budget=args.p_budget, dataset=args.dataset_name)
            for pim_i, pert_img in enumerate(perturbed_data):
                output_adv = model_best(pert_img).detach().to(device)
                accs_fgsm[pim_i] += float(metric(output_adv,labels))/(k*batch_size)

            loss_hist_test += float(loss_test.item())/(k*batch_size)
            acc_hist_test += float(metric(outputs,labels))/(k*batch_size)

        final_acc = accs_fgsm
        print('best_model acc robustness', final_acc)
        epoch_data = [epochs+1,model_best.cr,round(optimizer.integrator.param_groups[0]["lr"],5),-1,-1,-1,\
                        -1,round(acc_hist_test*100,4), final_acc,model_best.get_ranks(),model_best.get_conds(),model_best.get_mean_svs(),-1]

        running_data.loc[epochs+1] = epoch_data

        if save_name is not None and save_progress:
            running_data.to_csv(path+save_name+'.csv')

    return running_data
