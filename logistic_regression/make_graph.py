"""
Used by both logistic_regression and DeepInf
to generate a global graph object from the csv file
"""

from igraph import Graph
import pandas as pd

def make_graph(csvfile):

    edges = pd.read_csv(csvfile)

    edges["pid1s"] = edges.apply(lambda row: min(row["pid1"], row["pid2"]), axis=1)
    edges["pid2s"] = edges.apply(lambda row: max(row["pid1"], row["pid2"]), axis=1)
    
    edges = edges.drop(columns=['pid1', 'pid2'])
    edges = edges.rename(columns={'pid1s':'pid1', 'pid2s':'pid2'})

    collapsed_edges = edges[["pid1", "pid2", "duration"]].groupby(["pid1", "pid2"]).sum()
    
    return Graph.DataFrame(collapsed_edges.reset_index(), directed=False)
