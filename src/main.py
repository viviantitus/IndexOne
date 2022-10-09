import torch
from profiler import profileit
import argparse
import kmeans
import pq

parser = argparse.ArgumentParser(description='Vector DB')
parser.add_argument('-p', '--profile', dest='profile', action='store_true',
                    default=False,
                    help='to profile functions in the app')
args = parser.parse_args()


@profileit(enabled=args.profile)
def main():
    dataset = torch.cat([kmeans.get_dataset(100, 512) * 2, kmeans.get_dataset(100, 512)])
    codes, encoded_dataset = pq.train(dataset, num_chunks=32, num_centroids_per_chunk=8, chunk_dim=16)
    encoded_dataset = pq.encode(dataset, codes, 32, 8)
    distance_map = pq.compute_distance_map_for_centroids(codes, 32, 8)


    query = torch.stack([dataset[0, :], dataset[10, :]])
    encoded_queries = pq.encode(query, codes, 32, 8)
    pq.get_scores(encoded_dataset, encoded_queries, distance_map, 32)

main()