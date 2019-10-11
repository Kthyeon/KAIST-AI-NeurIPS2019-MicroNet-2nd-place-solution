import torch
from torch.optim import lr_scheduler
import copy
from torch import cuda, nn, optim
from tqdm import tqdm, trange
from Pruning import *
import numpy
from Utils.cutmix import rand_bbox
from torch.nn.functional import normalize
from Regularization import *
from Utils.utils import accuracy, AverageMeter, progress_bar, get_output_folder
import time
from torch.nn.utils import clip_grad_norm_
    


def train_32bit(model, dataloader, test_loader, args):
    device = model.device
    momentum = model.momentum
    learning_rate = model.lr
    num_epochs = model.num_epochs
    milestones = model.milestones
    gamma = model.gamma
    weight_decay = model.weight_decay
    nesterov = model.nesterov
    if args.label_regularize == 'labelsmooth':
        criterion = LabelSmoothing(model.device, model.num_classes, 0.1, 1)
    else:
        criterion = model.criterion    
    batch_number = len(dataloader.dataset) // dataloader.batch_size
    
    if args.batch_wd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov,
                                weight_decay=weight_decay)
    else:
        batch_params = [module for module in model.parameters() if module.ndimension() == 1]
        other_params = [module for module in model.parameters() if module.ndimension() > 1]
        optimizer = torch.optim.SGD([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': weight_decay}], lr=learning_rate, momentum=momentum, nesterov=nesterov)
    
    if args.lr_type == 'step':
        scheduler = lr_scheduler.MultiStepLR(gamma=gamma, milestones=milestones, optimizer=optimizer)
    elif args.lr_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = num_epochs, eta_min=0.0005, last_epoch=-1)
    losses = []
    test_losses = []
    accuracies = []
    test_accuracies = []
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        if args.label_regularize == 'labelsimilar':
            similarity = fc_similarity(model, device)
            criterion = LabelSimilarLoss(model.device, model.num_classes, similarity, 0.3, 1)
        
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            if args.input_regularize:
                if args.input_regularize == 'cutmix':
                    lam, images, labels_a, labels_b = cutmix_32bit(images, labels, device)
                elif args.input_regularize == 'mixup':
                    lam, images, labels_a, labels_b = mixup_32bit(images, labels, device)
                optimizer.zero_grad()

                outputs = model(images)

                loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            if args.ortho:
                loss += args.ortho_lr * l2_reg_ortho_32bit(model, device)
                

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            if (i + 1) % (batch_number // 4) == 0:
                tqdm.write('Epoch[{}/{}] , Step[{}/{}], Loss: {:.4f}, lr = {}'.format(epoch + 1,
                                                                                                          num_epochs,
                                                                                                          i + 1, len(
                        dataloader), loss.item(), optimizer.param_groups[0]['lr']))
            
            
        print('|| Test : Epoch {} / {} ||'.format(epoch, num_epochs))
        test_accuracy, test_loss = eval_32bit(model, test_loader)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

    return losses, accuracies, test_losses, test_accuracies, best_model_wts

def eval_32bit(model, test_loader):
    device = model.device
    criterion = model.criterion
    
    model.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    
    for i, data in enumerate(tqdm(test_loader)):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        pred_label = model(image)

        loss = criterion(pred_label, label)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred_label.data, label.data, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        top5.update(prec5.item(), image.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

    print('Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'.format(losses.avg, top1.avg, top5.avg))

    acc = top1.avg
    loss = losses.avg

    return acc, loss

def train_prune_32bit(model, dataloader, test_loader, best_model_wts_init, args, prune_rate = 50.):
    device = model.device
    momentum = model.momentum
    learning_rate = model.lr
    num_epochs = model.num_epochs
    milestones = model.milestones
    gamma = model.gamma
    weight_decay = model.weight_decay
    nesterov = model.nesterov
    
    if args.label_regularize == 'labelsmooth':
        criterion = LabelSmoothingLoss(model.device, model.num_classes, 0.1, 1)
    else:
        criterion = model.criterion
    batch_number = len(dataloader.dataset) // dataloader.batch_size
    
    if args.batch_wd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov,
                                weight_decay=weight_decay)
    else:
        batch_params = [module for module in model.parameters() if module.ndimension() == 1]
        other_params = [module for module in model.parameters() if module.ndimension() > 1]
        optimizer = torch.optim.SGD([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': weight_decay}], lr=learning_rate, momentum=momentum, nesterov=nesterov)
        
    if args.lr_type == 'step':
        scheduler = lr_scheduler.MultiStepLR(gamma=gamma, milestones=milestones, optimizer=optimizer)
    elif args.lr_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = num_epochs, eta_min=0.0005, last_epoch=-1)
    losses = []
    test_losses = []
    accuracies = []
    test_accuracies = []
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    ortho_miles = []
    
    for epoch in range(num_epochs):
        #get mask
        if epoch ==0:
            masks = weight_prune(model, prune_rate)
            model.load_state_dict(best_model_wts_init, strict = False)
            model.set_masks(masks)
        model.train()
        if args.label_regularize == 'labelsimilar':
            similarity = fc_similarity(model, device)
            criterion = LabelSimilarLoss(model.device, model.num_classes, similarity, 0.1, 1)
        scheduler.step()

        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            if args.input_regularize:
                if args.input_regularize == 'cutmix':
                    lam, images, labels_a, labels_b = cutmix_32bit(images, labels, device)
                elif args.input_regularize == 'mixup':
                    lam, images, labels_a, labels_b = mixup_32bit(images, labels, device)
                optimizer.zero_grad()

                outputs = model(images)

                loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            if args.ortho:
                loss += args.ortho_lr * l2_reg_ortho_32bit(model, device)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            if (i + 1) % (batch_number // 4) == 0:
                tqdm.write('Epoch[{}/{}] , Step[{}/{}], Loss: {:.4f}, lr = {}'.format(epoch + 1,
                                                                                                          num_epochs,
                                                                                                          i + 1, len(
                        dataloader), loss.item(), optimizer.param_groups[0]['lr']))
        #print('|| Train || === ', end = '')
        model.set_masks(masks)
        #tr_accuracy, tr_loss = eval_16bit(model, dataloader)
        print('|| Test  || === ', end = '')
        test_accuracy, test_loss = eval_32bit(model, test_loader)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
       # accuracies.append(tr_accuracy)
        #losses.append(tr_loss)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

    return losses, accuracies, test_losses, test_accuracies, best_model_wts

