import torch
import kmeans


def split_vec_into_chunks(dataset):
    for indx_ in range(int(dataset.size()[1]/16)):
        yield dataset[:, 16*indx_: 16*(indx_+1)]

def product_quantize(dataset):
    indxpq = []
    code_book = []
    for indx_, sub_vec in enumerate(split_vec_into_chunks(dataset)):
        _, centroids, assignements = kmeans.compute_kmeans(sub_vec, 8, 30)
        code_book.append(centroids)
        indxpq.append(assignements)

    return torch.stack(indxpq), torch.stack(code_book)


dataset = torch.cat([kmeans.get_dataset(100, 512) * 2, kmeans.get_dataset(100, 512)])
indxpq, codebook = product_quantize(dataset)
print(codebook.size())

#TODO: match one query to codebook and get ranking one
