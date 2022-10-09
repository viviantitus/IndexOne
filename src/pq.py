import torch
import kmeans
from distance import compute_distance


def split_to_subvecs(feature_vectors, num_chunks):
    for indx_ in range(num_chunks):
        yield feature_vectors[:, 16*indx_: 16*(indx_+1)]

def train(feature_vectors, num_chunks, num_centroids_per_chunk, chunk_dim):
    codes = torch.zeros((num_chunks, num_centroids_per_chunk, chunk_dim))
    assignments = torch.zeros((feature_vectors.size()[0], num_chunks))
    for i, sub_vec in enumerate(split_to_subvecs(feature_vectors, num_chunks=num_chunks)):
        _, codes[i], assignments[:, i] = kmeans.compute_kmeans(sub_vec, num_centroids=num_centroids_per_chunk, num_iterations=30)
    return codes, assignments

def compute_distance_map_for_centroids(codes, num_chunks, num_centroids_per_chunk):
    distance_map = torch.zeros((num_chunks, num_centroids_per_chunk, num_centroids_per_chunk))
    for indx_ in range(num_chunks):
        for i in range(num_centroids_per_chunk):
            distance_map[indx_, :, :] = compute_distance(codes[indx_, :, :], codes[indx_, :, :])
    return distance_map

def encode(feature_vectors, codes, num_chunks, num_centroids_per_chunk):
    feature_vectors_len = feature_vectors.size()[0]
    encoded_queries = torch.zeros((feature_vectors_len, num_chunks, num_centroids_per_chunk))

    for i, sub_vec in enumerate(split_to_subvecs(feature_vectors, num_chunks=num_chunks)):
        encoded_queries[:, i, :] = compute_distance(sub_vec, codes[i, :, :])
    
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

def adc_pq(query, encoded_dataset, codes, num_chunks, chunk_dim):
    encoded_dataset_len = encoded_dataset.size()[0]
    query_len = query.size()[0]
    distances = torch.zeros((query_len, encoded_dataset_len, num_chunks))

    for k, sub_vec in enumerate(split_to_subvecs(query, num_chunks=num_chunks)):
        for i in range(encoded_dataset_len):
            distances[:, i, k] = compute_distance(codes[k, encoded_dataset[i, k], :].unsqueeze(0), sub_vec).squeeze(0)

    return torch.sum(distances, dim=2)


