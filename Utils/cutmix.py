import torch
import torch.nn as nn
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mix_image(lam, x1, x2):
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
    
    return x1