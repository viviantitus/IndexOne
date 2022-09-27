import torch
from profiler import profileit

def get_rand_query(dimension: int = 512):
    return torch.rand(size=(dimension,))

def dataset_generator(dataset_size, dimension: int = 512):
    for i in range(dataset_size):
        yield torch.rand(size=(dimension,))

def cosine_similarity(x, y):
    dot_product = torch.dot(x, y)
    norms = torch.norm(x) * torch.norm(y)
    return dot_product/ (norms + 1e-6)

@profileit
def main():
    query = get_rand_query()
    for sample in dataset_generator(1000):
        print(cosine_similarity(query, sample))

main()