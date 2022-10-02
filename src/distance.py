import torch

def cosine_similarity(q, dataset):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    return 1. - torch.stack([cos(q[i], dataset) for i in range(q.size()[0])])

def compute_distance(q, dataset, type: str = "cosine"):
    if type == "cosine":
        return cosine_similarity(q, dataset)
    else:
        raise NotImplementedError("{} method not implemented ".format(type))