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
    # dataset = kmeans.get_dataset(100000,512)
    # codes, encoded_dataset = pq.train(dataset, num_chunks=32, num_centroids_per_chunk=8, chunk_dim=16)

    # query = torch.stack([dataset[3001, :], dataset[18005, :]])
    # distances = pq.adc_pq(query, encoded_dataset, codes, 32, 16)
    # print(torch.topk(distances, k=5, largest=False, dim=1).indices)

    from sklearn.cluster import KMeans
    import time
    import numpy as np
    X = np.random.rand(1000000,512).astype(np.float32)

    kmeans = KMeans(n_clusters=3, init='random')
    start = time.time()
    kmeans.fit(X)
    end = time.time()
    print(end - start)


main()
