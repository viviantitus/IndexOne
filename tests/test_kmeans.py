import torch
from distance import compute_distance
from kmeans import get_random_samples_from_dataset, get_dataset


def test_assignments():
    dataset = get_dataset(3)
    query = torch.stack([dataset[i, :] for i in range(3)])
    distances = compute_distance(query, dataset, type="cosine")
    assignments = torch.argmin(distances, dim=0)
    assert assignments[0] == 0 and assignments[1] == 1 and assignments[2] == 2