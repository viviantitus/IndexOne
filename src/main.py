import torch
from profiler import profileit


def get_rand_query(dimension: int = 512):
    return torch.rand(size=(1, dimension))

def dataset_generator(dataset_size, dimension: int = 512):
    default_size = 1000000
    for i in range(int(dataset_size/default_size)):
        yield torch.rand(size=(dimension, default_size))

def similiarity(q, dataset):
    num = torch.mm(q, dataset)
    den = torch.norm(q) * torch.norm(dataset, dim=0)
    return torch.flatten(num)/den

@profileit
def main(dataset_size):
    q = get_rand_query()
    for dataset in dataset_generator(dataset_size):
        similiarity(q, dataset)


     
main(dataset_size = 10000000)