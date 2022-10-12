import torch


def get_rand_query(dimension: int = 512):
    return torch.rand(size=(1, dimension))

def slice_func(big_number, slice_number):
    gen_list = [slice_number for i in range(int(big_number/slice_number))]
    if big_number % slice_number != 0:
        gen_list.append(big_number % slice_number)
    return gen_list

def dataset_generator(dataset_size: int, dimension: int = 512):
    for slice_size in slice_func(dataset_size, 1000000):
        yield torch.rand(size=(slice_size, dimension))