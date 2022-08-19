"""
Used by both logistic_regression and DeepInf
to generate a global graph object from the csv file
"""

from igraph import Graph
import pandas as pd

def make_graph(csvfile):

    edges = pd.read_csv(csvfile)

    edges["pid1"] = edges.apply(lambda row: min(row["sourcePID"], row["targetPID"]), axis=1)
    edges["pid2"] = edges.apply(lambda row: max(row["sourcePID"], row["targetPID"]), axis=1)

    collapsed_edges = edges[["pid1", "pid2", "duration"]].groupby(["pid1", "pid2"]).sum()

    return Graph.DataFrame(collapsed_edges.reset_index(), directed=False)
