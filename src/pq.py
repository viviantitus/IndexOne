import torch
import kmeans
from distance import compute_distance


def split_vec_into_chunks(dataset):
    for indx_ in range(int(dataset.size()[1]/16)):
        yield dataset[:, 16*indx_: 16*(indx_+1)]

def train_and_encode(dataset):
    encoded_dataset = []
    codes = []
    for _, sub_vec in enumerate(split_vec_into_chunks(dataset)):
        _, centroids, assignements = kmeans.compute_kmeans(sub_vec, 8, 30)
        codes.append(centroids)
        encoded_dataset.append(assignements)

    return torch.stack(encoded_dataset, dim=1), torch.stack(codes)

def compute_distance_map_for_centroids(codes, num_chunks, num_centroids_per_chunk):
    distance_map = torch.zeros((num_chunks, num_centroids_per_chunk, num_centroids_per_chunk))
    for indx_ in range(num_chunks):
        for i in range(num_centroids_per_chunk):
            distance_map[indx_, :, :] = compute_distance(codes[indx_, :, :], codes[indx_, :, :])
    return distance_map

def encode(queries, codes, num_chunks, num_centroids_per_chunk):
    query_len = queries.size()[0]
    encoded_queries = torch.zeros((query_len, num_chunks, num_centroids_per_chunk))

    for indx_ in range(num_chunks):
        encoded_queries[:, indx_, :] = compute_distance(queries[:, 16*indx_: 16*(indx_+1)], codes[indx_, :, :])
    
    return torch.min(encoded_queries, dim=2).indices

def get_scores(encoded_dataset, encoded_queries, distance_map, num_chunks):
    len_dataset = encoded_dataset.size()[0]
    len_queries = encoded_queries.size()[0]
    scores = torch.zeros((len_queries, len_dataset))

    for chunk_indx_ in range(num_chunks):
        for i in range(len_queries):
            for j in range(len_dataset):
                scores[i, j] = torch.sum(distance_map[:, encoded_dataset[j, chunk_indx_], encoded_queries[i, chunk_indx_]])
    return scores

dataset = torch.cat([kmeans.get_dataset(100, 512) * 2, kmeans.get_dataset(100, 512)])
encoded_dataset, codes = train_and_encode(dataset)
distance_map = compute_distance_map_for_centroids(codes, 32, 8)


query = torch.stack([dataset[0, :], dataset[101, :]])
encoded_queries = encode(query, codes, 32, 8)
print(torch.topk(get_scores(encoded_dataset, encoded_queries, distance_map, 32), dim=1, largest=False, k=15).indices)

