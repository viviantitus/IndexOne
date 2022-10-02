from random import randint
import torch
from distance import compute_distance
from data_gen import dataset_generator


def get_dataset(size: int):
    for dataset in dataset_generator(size):
        return dataset


def get_random_samples_from_dataset(dataset: torch.Tensor, num_centroids: int):
        random_indices = []
        while len(random_indices) < num_centroids:
            x = randint(0, dataset.size()[0]-1)
            if x not in random_indices:
                random_indices.append(x)
        return torch.stack([dataset[i, :] for i in random_indices])


def reassign_centroids(dataset: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor):
    new_centroids = []
    for i in range(centroids.size()[0]):
        new_centroids.append(torch.mean(dataset[assignments == i, :], dim=0))
    return torch.stack(new_centroids)


def calculate_variance(dataset: torch.Tensor, assignments: torch.Tensor, centroids: torch.Tensor):
    variance = 0
    for i in range(centroids.size()[0]):
        samples = dataset[assignments == i, :]
        centroid = centroids[i].unsqueeze(0).repeat(samples.size()[0], 1)
        variance += torch.norm(samples - centroid)
    return variance


def compute_kmeans(dataset: torch.Tensor, num_centroids, num_iterations):    
    ret_centroids = None
    ret_assignments = None
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
                    ret_centroids = centroids
                    min_variance = curr_variance
                    ret_assignments = assignments
                    print(variance)
                break
            variance = curr_variance
            centroids = reassign_centroids(dataset, assignments, centroids)
    return (ret_centroids, ret_assignments)

dataset = get_dataset(1000)
centroids, assignments = compute_kmeans(dataset, 4, 1000)
print(assignments)