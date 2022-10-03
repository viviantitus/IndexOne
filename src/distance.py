import torch

def cosine_distance(q, dataset):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    return 1. - torch.stack([cos(q[i], dataset) for i in range(q.size()[0])])

def euclidean_distance(q, dataset):
    return torch.stack([torch.linalg.norm(q[i].unsqueeze(0).repeat(dataset.size()[0], 1) - dataset, dim=1) for i in range(q.size()[0])])

def dotprod_distance(q, dataset):
    mm = torch.mm(dataset, q.transpose(0,1))
    return - mm.transpose(0, 1)

def compute_distance(q, dataset, type: str = "cosine"):
    if type == "cosine":
        return cosine_distance(q, dataset)
    elif type == "euclidean":
        return euclidean_distance(q, dataset)
    elif type == "dotprod":
        return dotprod_distance(q, dataset)
    else:
        raise NotImplementedError("{} method not implemented ".format(type))