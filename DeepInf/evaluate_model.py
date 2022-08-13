from argparse import ArgumentParser
import os
import numpy as np
import torch
import pandas as pd

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from DeepInfSrc.gat import BatchGAT
from DeepInfSrc.utils import load_w2v_feature
from DeepInfSrc.data_loader import InfluenceDataSet, ChunkSampler

if __name__ == "__main__":
    
    parser = ArgumentParser(description="Take a model and turn predictions into a csv")
    parser.add_argument("model_file", help="torch model file")
    parser.add_argument("training_dir", help="training_dir")
    parser.add_argument("output_file", help="output file")
    parser.add_argument("--max-vertex-id", type=int, help="max vertex index")
    parser.add_argument("--eval-labels", help="evaluation label csv")
    args = parser.parse_args()
    data = InfluenceDataSet(
        args.training_dir,
        64,
        000,
        False,
        "gat",
        args.max_vertex_id
    )

    loader = DataLoader(data, batch_size=1024, 
                        sampler=ChunkSampler(len(data),0))
    # will need to be in the actual directory, I guess?
    
    n_units = [data.get_feature_dimension(), 16, 16, 2]
    try:
        model = BatchGAT(pretrained_emb=data.get_embedding(), 
                    vertex_feature=data.get_vertex_features(), 
                    use_vertex_feature=True, 
                    n_units=n_units, n_heads=[8,8,1], 
                    dropout=0.2, instance_normalization=True)
        model.load_state_dict(torch.load(args.model_file, map_location=torch.device("cpu"))) 
    except RuntimeError:
        model = BatchGAT(data.get_embedding(), data.get_vertex_features(), False, 
                    n_units=n_units, n_heads=[8,8,1], 
                    dropout=0.2, instance_normalization=True)
        model.load_state_dict(torch.load(args.model_file, map_location=torch.device("cpu"))) 
    model.eval()

    dfs = []
 
    for batch in loader:
     
        batch_graph, batch_feat, batch_label, batch_vert = batch
     
        output = model(batch_feat, batch_vert, batch_graph)[:,-1,:]
        y_pred = output.max(1)[1].data.tolist()
        log_y0 = output[:,0].data.tolist()
        log_y1 = output[:,1].data.tolist()

        y0 = np.exp(log_y0)
        y1 = np.exp(log_y1)
        # pick out pid 
        indicator = batch_feat[:,:,0] == 1.0     
        pid_values = batch_vert[indicator]
    
        batch_df = pd.DataFrame({"pid" : pid_values, "y_pred" : y_pred,
                      "y0" : y0, "y1" : y1, "s_t+d" : batch_label})
        dfs.append(batch_df)
    out_df = pd.concat(dfs)
    out_df.to_csv(args.output_file)

    eval_labels = pd.read_csv(args.eval_labels)
    eval_labels.set_index("pid", inplace=True)

    eval_preds = out_df[["pid", "y1"]].groupby(["pid"]).max()

    eval_data = eval_labels.join(eval_preds).fillna(0.0)

    gnn_auprc = average_precision_score(eval_data["infected"], eval_data["y1"])
    
    print(f"GNN AUPRC = {gnn_auprc}")
