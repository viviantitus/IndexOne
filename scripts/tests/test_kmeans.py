import torch
from distance import compute_distance
from kmeans import get_random_samples_from_dataset, get_dataset


def test_assignments_with_cosine_distance():
    dataset = get_dataset(3, 512)
    query = torch.stack([dataset[i, :] for i in range(3)])
    distances = compute_distance(query, dataset, type="cosine")
    assignments = torch.argmin(distances, dim=0)
    assert assignments[0] == 0 and assignments[1] == 1 and assignments[2] == 2


def test_assignments_with_euclidean_distance():
    dataset = get_dataset(3, 512)
    query = torch.stack([dataset[i, :] for i in range(3)])
    distances = compute_distance(query, dataset, type="euclidean")
    assignments = torch.argmin(distances, dim=0)
    assert assignments[0] == 0 and assignments[1] == 1 and assignments[2] == 2

def test_assignments_with_dotprod_distance():
    dataset = get_dataset(3, 512)
    query = torch.stack([dataset[i, :] for i in range(3)])
    distances = compute_distance(query, dataset, type="dotprod")
    assignments = torch.argmin(distances, dim=0)
    assert assignments[0] == 0 and assignments[1] == 1 and assignments[2] == 2
