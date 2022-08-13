# merge train and test embeddings 
# prefer test embeddings over train embeddings
import numpy as np
import os
from pickle import dump
from argparse import ArgumentParser

from collate import merge_vertex_features

def parse_emb(emb_file):
    output = {}
    with open(emb_file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            if nu == 0:
                n, d = int(content[0]), int(content[1])
                nu += 1
                continue
            else:
                index = int(content[0])
                feature = np.zeros((d,))
                for i,x in enumerate(content[1:]):
                    feature[i] = float(x)
                output[index] = feature
            nu += 1
    return output

def merge_emb(emb_dict_1, emb_dict_2):
    """
    Take all entries in dict 2 not in dict 1 and 
    add them to dict_1
    """
    emb_dict_2.update(emb_dict_1)
    return emb_dict_2

def output_emb(emb_dict, outfile):
    n = max(emb_dict.keys())
    d = len(emb_dict[n])

    ordered_keys = list(emb_dict.keys())
    ordered_keys.sort()

    with open(outfile, "w") as f:
        f.write(f"{n} {d}\n")
        for k in ordered_keys:
            str_out = np.array_str(emb_dict[k])[1:-1].replace("\n", "")
            f.write(f"{k} {str_out}\n")

def compress_vertices(emb_dict, features, *vertex_ids):
    """
    produce a mapping between old expanded vertices
    and only those used in embedding and in features
    """
    non_zero_emb = list(emb_dict.keys())
    non_zero_feature = np.where((features != 0.0).any(axis=1))[0]
    
    print(f"There are {len(non_zero_emb)} non-zero embedding nodes and {len(non_zero_feature)} non-zero feature nodes")
    used_vertices = set(non_zero_emb).intersection(set(non_zero_feature))
     
    id_map = {pid : idx for idx,pid in enumerate(used_vertices)}
    rev_id = {idx : pid for pid,idx in id_map.items()}
    
    lookup_func = np.vectorize(lambda x: id_map.get(x, len(id_map)))
    
    new_emb = {}
    new_features = np.zeros((len(id_map)+1, features.shape[1]))
    new_vertex_ids = (lookup_func(vertex_id) for vertex_id in vertex_ids)
    
    for i in range(len(id_map)):
        new_emb[i] = emb_dict[rev_id[i]]
        new_features[i] = features[rev_id[i]]
    return new_emb, new_features, rev_id, *new_vertex_ids

if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("train_dir")
    parser.add_argument("test_dir")
    parser.add_argument("output_train")
    parser.add_argument("output_test")

    args = parser.parse_args()

    train_emb = parse_emb(os.path.join(args.train_dir, "deepwalk.emb_64.part"))
    test_emb = parse_emb(os.path.join(args.test_dir, "deepwalk.emb_64.part"))

    test_emb.update(train_emb)

    # merge and expand vertex features
    train_feature = np.load(os.path.join(args.train_dir, "vertex_feature.npy"))
    test_feature = np.load(os.path.join(args.test_dir, "vertex_feature.npy"))

    merged_vertex = merge_vertex_features([train_feature, test_feature])

    vertex_id_train = np.load(os.path.join(args.train_dir, "vertex_id.npy"))
    vertex_id_test = np.load(os.path.join(args.test_dir, "vertex_id.npy"))

    # do deduplication
    new_emb, new_features, rev_id, new_vertex_train, new_vertex_test = compress_vertices(
        test_emb, merged_vertex, vertex_id_train, vertex_id_test)

    with open(os.path.join(args.output_test, "rev_id.dict"), "wb") as id_file:
        dump(rev_id, id_file)
    output_emb(new_emb, os.path.join(args.output_train, "deepwalk.emb_64"))
    output_emb(new_emb, os.path.join(args.output_test, "deepwalk.emb_64"))

    np.save(os.path.join(args.output_train, "vertex_feature.npy"), new_features)
    np.save(os.path.join(args.output_test, "vertex_feature.npy"), new_features)

    np.save(os.path.join(args.output_train, "vertex_id.npy"), new_vertex_train)
    np.save(os.path.join(args.output_test, "vertex_id.npy"), new_vertex_test)
