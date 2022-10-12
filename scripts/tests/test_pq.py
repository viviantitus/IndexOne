import torch
from pq import compute_distance_map_for_centroids


def test_compute_distance_map_for_centroids():
    codes = torch.rand((32, 8, 16))
    distance_map = compute_distance_map_for_centroids(codes, num_chunks=32, num_centroids_per_chunk=8)
    assert distance_map.size() == torch.zeros((32, 8, 8)).size() and torch.sum(torch.diagonal(distance_map, dim1=1, dim2=2)) < 1e-6
