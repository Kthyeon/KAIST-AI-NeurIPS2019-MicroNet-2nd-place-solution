import torch
import copy
from torch import cuda, nn
import numpy



class LabelSmoothingLoss(nn.Module):
    def __init__(self, device, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.device = device

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).to(self.device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
def fc_similarity(model, device):
    weight_matrix = model.linear.weight.data
    num = weight_matrix.shape[0]
    cos_si = nn.CosineSimilarity(dim=0)
    similarity = torch.zeros([num,num]).to(device)
    for i in range(num):
        for j in range(num):
            similarity[i,j] = torch.abs(cos_si(weight_matrix[i], weight_matrix[j]))
        similarity[i,i] = 0.
        similarity[i] = similarity[i]/torch.sum(similarity[i])
    
    return similarity
    
class LabelSimilarLoss(nn.Module):
    def __init__(self, device, classes, similarity, smoothing=0.0, dim=-1):
        super(LabelSimilarLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.device = device
        self.similarity = similarity
        #similarity
        #diagonal: 0, w_ij = fi*fj/Z (fi,j: unit fully connected vector, Z: normalize constant on row i)

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).to(self.device)
            for i in range(target.shape[0]):
                true_dist[i] = self.similarity[target[i]] * self.smoothing
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))