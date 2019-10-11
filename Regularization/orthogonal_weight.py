import torch
from torch.optim import lr_scheduler
import copy
from torch import cuda, nn, optim
from tqdm import tqdm, trange
import numpy
from torch.nn.functional import normalize
from torch.autograd import Variable

def gram_schmidt(vectors,device):
    #to make pointwise matrix independent matrix
    if vectors.shape[0]>vectors.shape[1]:
        vectors = vectors.transpose(0,1)
    basis = torch.zeros_like(vectors).to(device)
    for num in range(vectors.shape[0]):
        temp = torch.zeros_like(vectors[num])
        for b in basis:
            temp += torch.matmul(vectors[num],b) * b
        w = vectors[num] - temp
        if (w > 1e-10).any():  
            basis[num] = w/torch.norm(w)
    basis = basis.half()
    
    if vectors.shape[0]>vectors.shape[1]:
        return basis.transpose(0,1) 
    else:
        return basis
    
def gr_sch_pr(mdl, device):
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                m.conv1.weight = torch.nn.Parameter(gram_schmidt(m.conv1.weight, device))
                m.conv3.weight = torch.nn.Parameter(gram_schmidt(m.conv3.weight, device))
    
    

    
def l2_reg_ortho(mdl, device):
    """
    SRIP function from 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
    https://arxiv.org/abs/1810.09102.
    """
    l2_reg = None
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                W = m.conv1.weight
                cols = W[0].numel()
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                m  = torch.matmul(wt,w1)
                ident = Variable(torch.eye(cols,cols)).type(torch.HalfTensor).to(device)

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (sigma)**2
                else:
                    l2_reg = l2_reg + (sigma)**2
                    
                
    return l2_reg
'''
def l2_reg_ortho_32bit(mdl, device):
    """
    SRIP function from 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
    https://arxiv.org/abs/1810.09102.
    """
    l2_reg = None
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                W = m.conv1.weight
                cols = W[0].numel()
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                m  = torch.matmul(wt,w1)
                ident = Variable(torch.eye(cols,cols)).to(device)

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (sigma)**2
                else:
                    l2_reg = l2_reg + (sigma)**2
                    
                
    return l2_reg
'''
def l2_reg_ortho_32bit(mdl, device):
    l2_reg = None
    for W in mdl.parameters():
        if W.ndimension() < 2:
            continue
        elif W.ndimension() == 4:
            if W.shape[3] == 1:
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                m  = torch.matmul(wt,w1)
                ident = Variable(torch.eye(cols,cols)).to(device)            

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (torch.norm(sigma,2))**2
                    num = 1
                else:
                    l2_reg = l2_reg + (torch.norm(sigma,2))**2
                    num += 1
                    
    return l2_reg / num

def conv3_l2_reg_ortho(mdl, device):
    """
    SRIP function from 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
    https://arxiv.org/abs/1810.09102.
    """
    l2_reg = None
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                W = m.conv3.weight
                cols = W[0].numel()
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                m  = torch.matmul(wt,w1)
                ident = Variable(torch.eye(cols,cols)).type(torch.HalfTensor).to(device)

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (sigma)**2
                else:
                    l2_reg = l2_reg + (sigma)**2
                    
                
    return l2_reg

def conv3_l2_reg_ortho_32bit(mdl, device):
    """
    SRIP function from 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
    https://arxiv.org/abs/1810.09102.
    """
    l2_reg = None
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                W = m.conv3.weight
                cols = W[0].numel()
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                m  = torch.matmul(wt,w1)
                ident = Variable(torch.eye(cols,cols)).to(device)

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if l2_reg is None:
                    l2_reg = (sigma)**2
                else:
                    l2_reg = l2_reg + (sigma)**2
                    
                
    return l2_reg

def fc_l2_reg_ortho(mdl, device):
    """
    SRIP function from 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
    https://arxiv.org/abs/1810.09102.
    """
    l2_reg = None
    for name, m in mdl.named_children():
        if 'last' in name:
            W = m.weight
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1,cols)
            wt = torch.transpose(w1,0,1)
            m  = torch.matmul(wt,w1)
            ident = Variable(torch.eye(cols,cols)).type(torch.HalfTensor).to(device)

            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))
            
            if l2_reg is None:
                l2_reg = (sigma)**2
            else:
                l2_reg = l2_reg + (sigma)**2
                    
                
    return l2_reg

def conv1_l2_reg_orthogonal(mdl, device):
    """
    Make weight matrixs be an orthogonal matrix. (not a orthonormal matrix.)
    This is to analyze only the effect of orthogonality, not from the orthonormal vectors.
    """
    l2_reg = None
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                W = m.conv1.weight
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                if (rows > cols):
                    m  = torch.matmul(wt,w1)
                else:
                    m = torch.matmul(w1,wt)

                w_tmp = (m - torch.diagflat(torch.diagonal(m)))
                b_k = Variable(torch.rand(w_tmp.shape[1],1)).type(torch.HalfTensor).to(mdl.device)
                b_k = b_k.to(mdl.device)

                v1 = torch.matmul(w_tmp, b_k)
                norm1 = torch.norm(v1,2)
                v2 = torch.div(v1,norm1)
                v3 = torch.matmul(w_tmp,v2)

                if l2_reg is None:
                    l2_reg = (torch.norm(v3,2))**2
                else:
                    l2_reg = l2_reg + (torch.norm(v3,2))**2
                    
                
    return l2_reg

def conv3_l2_reg_orthogonal(mdl, device):
    """
    Make weight matrixs be an orthogonal matrix. (not a orthonormal matrix.)
    This is to analyze only the effect of orthogonality, not from the orthonormal vectors.
    """
    l2_reg = None
    for name, module in mdl.named_children():
        if 'layer' in name:
            for m in module:
                W = m.conv3.weight
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1,cols)
                wt = torch.transpose(w1,0,1)
                if (rows > cols):
                    m  = torch.matmul(wt,w1)
                else:
                    m = torch.matmul(w1,wt)

                w_tmp = (m - torch.diagflat(torch.diagonal(m)))
                b_k = Variable(torch.rand(w_tmp.shape[1],1)).type(torch.HalfTensor).to(mdl.device)
                b_k = b_k.to(mdl.device)

                v1 = torch.matmul(w_tmp, b_k)
                norm1 = torch.norm(v1,2)
                v2 = torch.div(v1,norm1)
                v3 = torch.matmul(w_tmp,v2)

                if l2_reg is None:
                    l2_reg = (torch.norm(v3,2))**2
                else:
                    l2_reg = l2_reg + (torch.norm(v3,2))**2
                    
                
    return l2_reg