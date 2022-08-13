import numpy as np
from argparse import ArgumentParser

def gen_edges(matrix, vertex_ids):

    edges = matrix.nonzero()

    for e1, e2 in zip(edges[0], edges[1]):
        yield vertex_ids[e1], vertex_ids[e2]

def gen_graph(adj_mat, vertex_ids):

    n = adj_mat.shape[0]

    for i in range(n):
        for e1, e2 in gen_edges(adj_mat[i], vertex_ids[i]):
            yield e1, e2

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("adj_matrix_file")
    parser.add_argument("vertex_id_file")
    parser.add_argument("outfile")

    args = parser.parse_args()

    adj_mat = np.load(args.adj_matrix_file)
    print(f"Translating {adj_mat.shape[0]} adjacency matrices to edgelist")

    vertex_ids = np.load(args.vertex_id_file)

    assert adj_mat.shape[0] == vertex_ids.shape[0]
   
    with open(args.outfile, "w") as outfile: 
        for e1, e2 in gen_graph(adj_mat, vertex_ids):
            outfile.write(f"{e1} {e2}\n")
