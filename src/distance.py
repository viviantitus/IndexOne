import torch

def cosine_similarity(q, dataset):
    q = q.transpose(0, 1)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    return 1.0 - cos(q, dataset)

def cosine_similarity2(q, dataset):
    dataset = dataset.transpose(0, 1)
    return torch.mm(q, dataset)

def compute_distance(q, dataset, type: str = "cosine"):
    if type == "cosine":
        return cosine_similarity2(q, dataset)
    else:
        raise NotImplementedError("{} method not implemented ".format(type))