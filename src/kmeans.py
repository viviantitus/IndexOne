from random import randint
import math
import torch
from distance import compute_distance
from data_gen import dataset_generator


def get_dataset(size: int):
    for dataset in dataset_generator(size):
        return dataset


def get_random_samples_from_dataset(dataset: torch.Tensor, num_centroids: int):
        random_indices = []
        for _ in range(num_centroids):
            x = randint(0, dataset.size()[1]-1)
            if x not in random_indices:
                random_indices.append(x)
        return torch.stack([dataset[:, i] for i in random_indices])


def reassign_centroids(dataset: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor):
    new_centroids = []
    for i in range(centroids.size()[0]):
        new_centroids.append(torch.mean(dataset[:, assignments == i], dim=1))
    return torch.stack(new_centroids)


def calculate_variance(dataset: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor):
    variance = 0
    for i in range(centroids.size()[0]):
        samples = dataset[:, assignments == i]
        centroid = centroids[i].unsqueeze(0).repeat(samples.size()[1], 1).transpose(0, 1)
        variance += torch.norm(samples - centroid)
    return variance


def compute_kmeans(num_centroids=3, num_iterations = 30):
    dataset = get_dataset(100)
    
    last_centroids = None
    min_variance = float('inf')
    for _ in range(num_iterations):
        centroids = get_random_samples_from_dataset(dataset, num_centroids)
        variance = min_variance
        while True:
            distances = compute_distance(centroids, dataset)
            assignments = torch.argmin(distances, dim=0)
            curr_variance = calculate_variance(dataset, assignments, centroids)
            if variance - curr_variance < 0.001:
                if min_variance > variance:
                    last_centroids = centroids
                    min_variance = curr_variance
                break
            variance = curr_variance
            centroids = reassign_centroids(dataset, assignments, centroids)
    return last_centroids

print(compute_kmeans())
