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
    codes, _ = pq.train(dataset, num_chunks=32, num_centroids_per_chunk=8, chunk_dim=16)
    encoded_dataset = pq.encode(dataset, codes, 32, 8)


    query = torch.stack([dataset[45, :], dataset[111, :]])
    distances = pq.adc_pq(query, encoded_dataset, codes, 32, 16)
    print(torch.topk(distances, k=5, largest=False, dim=1))


main()
