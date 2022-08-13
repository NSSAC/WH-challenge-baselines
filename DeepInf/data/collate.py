import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from glob import glob

# merge train and test embeddings 
# for node v, prefer embeddings where it was the ego network

def merge_vertex_features(vertex_features):

    max_n = max(vertex_feature.shape[0] for vertex_feature in vertex_features)
    full_vertex = np.zeros((max_n, 4))

    for vertex_feature in vertex_features:

        zero_in_full = np.argwhere((full_vertex == 0).all(axis=1)) # available rows
        non_zero_vertex = np.argwhere((vertex_feature != 0).any(axis=1)) # rows to write to
        
        conflicts = ~np.isin(non_zero_vertex, zero_in_full) 
        
        if conflicts.any():
            print(f"There are {conflicts.sum()}/{len(non_zero_vertex)} conflicts") 
            # verify ages are correct
            conflict_idx = tuple(non_zero_vertex[conflicts])
            old_ages = full_vertex[conflict_idx, 0]
            new_ages = vertex_feature[conflict_idx, 0]

            if (old_ages != new_ages).any():
                raise Exception(f"Conflicting age information old {old_ages} new {new_ages}")
        non_conflicted = tuple(non_zero_vertex[~conflicts]) 
        full_vertex[non_conflicted, :] = vertex_feature[non_conflicted, :]

    return full_vertex

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("directory", help="directory containing collected data")
    parser.add_argument("output_dir", help="final directory to place instances in")

    args = parser.parse_args()

    directories = glob(args.directory+"/data_*/")

    # we can stack adj_matrix, labels, vertex_id, inf_feature 
    # and for now ignoring vertex features
    label_list = []
    adj_list = []
    vertex_list = []
    inf_feature_list = []
    vertex_feat_list = []
    
    emb_lines = []

    for directory in directories:

        label_list.append(np.load(os.path.join(directory, "label.npy")))
        adj_list.append(np.load(os.path.join(directory, "adjacency_matrix.npy")))
        vertex_list.append(np.load(os.path.join(directory, "vertex_id.npy")))
        inf_feature_list.append(np.load(os.path.join(directory, "influence_feature.npy")))
        vertex_feat_list.append(np.load(os.path.join(directory, "vertex_features.npy")))

        with open(os.path.join(directory, "deepwalk.emb_64"), "r") as emb_file:
            emb_lines.extend(emb_file.readlines()[1:]) 

    labels = np.concatenate(label_list)
    adj = np.concatenate(adj_list)
    vertex_id = np.concatenate(vertex_list)
    inf_feature = np.concatenate(inf_feature_list)
    vertex_feat = merge_vertex_features(vertex_feat_list)

    print(f"There are {labels.sum()} positive instances to {len(labels)} total instances")

    np.save(os.path.join(args.output_dir, "label.npy"), labels)
    np.save(os.path.join(args.output_dir, "adjacency_matrix.npy"), adj)
    np.save(os.path.join(args.output_dir, "vertex_id.npy"), vertex_id)
    np.save(os.path.join(args.output_dir, "influence_feature.npy"), inf_feature)
    np.save(os.path.join(args.output_dir, "vertex_feature.npy"), vertex_feat) # yeah, I did it plural in the lower ones

    # need to do final collation of trainng and testing embedding
    with open(os.path.join(args.output_dir, "deepwalk.emb_64.part"), "w") as emb_out:
        emb_out.write(f"{len(emb_lines)} 64\n")
        for line in emb_lines:
            emb_out.write(line) 
