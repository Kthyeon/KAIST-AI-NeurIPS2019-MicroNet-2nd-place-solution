import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Pruning.utils import prune_rate, arg_nonzero_min


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(torch.abs(p).cpu().data.numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = torch.abs(p).data > threshold
            masks.append(pruned_inds.float())
    return masks

def per_layer_weight_prune(model, pruning_perc):
    '''
    On Progress...
    pruning_perc[0] : prune rate of other weights
    pruning_perc[1] : prune rate of point weights
    '''    
    other_weights = []
    point_weights = []
    
    for name, p in model.named_parameters():
        if len(p.data.size()) != 1:
            if 'layer' in name:
                if 'conv1' in name or 'conv3' in name:
                    point_weights += list(torch.abs(p).cpu().data.numpy().flatten())
                else:
                    other_weights += list(torch.abs(p).cpu().data.numpy().flatten())
            else:
                other_weights += list(torch.abs(p).cpu().data.numpy().flatten())
        

    threshold_other = np.percentile(np.array(other_weights), pruning_perc[0])
    threshold_point = np.percentile(np.array(point_weights), pruning_perc[1])
    
    num_of_params = [len(other_weights), len(point_weights)]
    # generate mask
    masks = []
    for name, p in model.named_parameters():
        if len(p.data.size()) != 1:
            if 'layer' in name:
                if 'conv1' in name or 'conv3' in name:
                    pruned_inds = torch.abs(p).data > threshold_point
                    masks.append(pruned_inds.float())
                else:
                    pruned_inds = torch.abs(p).data > threshold_other
                    masks.append(pruned_inds.float())
            else:
                pruned_inds = torch.abs(p).data > threshold_other
                masks.append(pruned_inds.float())
    
    return masks, num_of_params


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks
