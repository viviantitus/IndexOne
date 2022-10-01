import torch

def cosine_similiarity(q, dataset):
    num = torch.mm(q, dataset)
    den = torch.norm(q) * torch.norm(dataset, dim=0)
    return num/den

def compute_distance(q, dataset, type: str = "cosine"):
    if type == "cosine":
        return cosine_similiarity(q, dataset)
    else:
        raise NotImplementedError("{} method not implemented ".format(type))