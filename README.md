# IndexOne
Vector Database


Total time taken for each algorithm for computing distance of query to 1 billion vectors (512 dimensions)


Algorithm                    Time
cosine_similarity             264.32s
dot_product                   70.91s


Total time taken for each algorithm for computing distance of query to 1 billion vectors (128 dimensions)


Algorithm                    Time
cosine_similarity             81.20s
dot_product                   17.22s


Total time taken to compute kmeans is 69 seconds for 10000 vectors of dimension 512
Optimised version of rust takes 469ms which is 14612% increase whereas scikit python taken 1.36s