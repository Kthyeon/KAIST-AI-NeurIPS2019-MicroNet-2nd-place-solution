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
    


def train_32bit(model, dataloader, test_loader):
    device = model.device
    momentum = model.momentum
    learning_rate = model.lr
    num_epochs = model.num_epochs
    milestones = model.milestones
    gamma = model.gamma
    weight_decay = model.weight_decay
    nesterov = model.nesterov
    criterion = model.criterion

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov,
                                weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(gamma=gamma, milestones=milestones, optimizer=optimizer)
    losses = []
    test_losses = []
    accuracy = []
    test_accuracies = []
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    batch_number = len(dataloader.dataset) // dataloader.batch_size

    for epoch in range(num_epochs):
        model.train()
        scheduler.step()

        for i, (images, labels) in enumerate(dataloader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            if (i + 1) % (batch_number // 4) == 0:
                tqdm.write('Epoch[{}/{}] , Step[{}/{}], Loss: {:.4f}, lr = {}'.format(epoch + 1,
                                                                                                          num_epochs,
                                                                                                          i + 1, len(dataloader), loss.item(), optimizer.param_groups[0]['lr']))
        #print('|| Train || === ', end = '')
        #tr_accuracy, tr_loss = eval_32bit(model, dataloader)
        print('|| Test  || === ', end = '')
        test_accuracy, test_loss = eval_32bit(model, test_loader)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        #accuracy.append(tr_accuracy)
        #losses.append(tr_loss)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

    return losses, accuracy, test_losses, test_accuracies, best_model_wts

def train_mixup_32bit(model, dataloader, test_loader):
    device = model.device
    momentum = model.momentum
    learning_rate = model.lr
    num_epochs = model.num_epochs
    milestones = model.milestones
    gamma = model.gamma
    weight_decay = model.weight_decay
    nesterov = model.nesterov
    criterion = model.criterion
    pruning_time = model.pruning_time
    pruning_perc = model.pruning_perc
    batch_number = len(dataloader.dataset) // dataloader.batch_size

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov,
                                weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(gamma=gamma, milestones=milestones, optimizer=optimizer)
    losses = []
    test_losses = []
    accuracy = []
    test_accuracies = []
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        scheduler.step()

        for i, (images1, labels) in enumerate(dataloader2):
            images1 = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            lam, images, labels1, labels2 = mixup(images1, labels, device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = lam * criterion(outputs, labels1) + (1-lam) * criterion(outputs, labels2)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            if (i + 1) % (batch_number // 4) == 0:
                tqdm.write('Epoch[{}/{}] , Step[{}/{}], Loss: {:.4f}, lr = {}'.format(epoch + 1,
                                                                                                          num_epochs,
                                                                                                          i + 1, len(
                        dataloader), loss.item(), optimizer.param_groups[0]['lr']))
        print('|| Train || === ', end = '')
        tr_accuracy, tr_loss = eval_32bit(model, dataloader)
        print('|| Test  || === ', end = '')
        test_accuracy, test_loss = eval_32bit(model, test_loader)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        accuracy.append(tr_accuracy)
        losses.append(tr_loss)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

    return losses, accuracy, test_losses, test_accuracies, best_model_wts


def eval_32bit(model, test_loader):
    total = 0
    correct = 0
    device = model.device
    criterion = model.criterion

    model.eval()
    for i, data in enumerate(test_loader):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        pred_label = model(image)

        _, predicted = torch.max(pred_label.data, 1)
        total += label.shape[0]
        correct += (predicted == label).sum()
        if i == 0:
            loss = criterion(pred_label, predicted).item()
        else:
            loss += criterion(pred_label, predicted).item()

    accuracy = 100 * correct.item() / total
    tqdm.write('Accuracy : {}%, loss : {:.4f}'.format(accuracy, loss))

    return accuracy, loss

