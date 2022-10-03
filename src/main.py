import torch
from profiler import profileit
import argparse
import kmeans

parser = argparse.ArgumentParser(description='Vector DB')
parser.add_argument('-p', '--profile', dest='profile', action='store_true',
                    default=False,
                    help='to profile functions in the app')
args = parser.parse_args()


@profileit(enabled=args.profile)
def main():
    dataset = torch.cat([kmeans.get_dataset(100, 512) * 2, kmeans.get_dataset(100, 512)])
    variance, centroids, assignments = kmeans.compute_kmeans(dataset, num_centroids=2, num_iterations=1000, distance_type="euclidean")
    print(assignments.unique(return_counts=True))
    print(variance)

main()