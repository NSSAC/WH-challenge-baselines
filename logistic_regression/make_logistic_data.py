"""
Author: Galen Harrison
Date: 2022-08-11

This script generates features and labels for training
and the features for evaluating the logistic regression in 
logistic_regression.py

Note that this does not produce labels for evaluation, as
training and evaluation are contained in separate files
"""

import os
from math import ceil
from argparse import ArgumentParser
from glob import glob
from pickle import load

import pandas as pd
import numpy as np

from igraph import Graph

def get_subgraph_inf_weight(si_limit, subgraph, pid, steps):
    """
    Get the "infection weight" or the discounted weight of
    the shortest paths from all infected nodes (as defined by si_limit)
    to node specified by pid.

    Look out to maximum steps from pid

    si_limit - a DataFrame with "pid" as an index level representing infected
    pids within the relevant time window
   
    subgraph - an igraph Graph object
    
    pid - the pid of the vertex to compute for
    
    steps - the maximum number of steps to explore out from 
    """
  
    vid = subgraph.vs.find(name=pid)
    neighbors = subgraph.neighborhood(vid, order=steps)
    neighbor_pids = [subgraph.vs["name"][n] for n in neighbors]
    
    inf_neighbors = si_limit.index.get_level_values("pid").intersection(neighbor_pids)
    inf_vids = [subgraph.vs.find(name=pid) for pid in inf_neighbors]
    
    if len(inf_neighbors) == 0:
        return 0
    neighbor_edge_paths = subgraph.get_shortest_paths(
            vid,
            to=inf_vids,
            output="epath"
    )
    
    weight = 0
    for edge in neighbor_edge_paths:
        coeff = 1.0
        decay_term = 0.5
        
        for e in edge:
            weight += coeff*subgraph.es[e]["duration"]
            coeff = decay_term*coeff
    return weight

def make_si_table(disease_data):
    """
    make table of all infections indexed by pid and infected (day of infection)
    """
    is_infected = disease_data["state"] == "I"
    pid = disease_data[is_infected]["pid"]
    day = disease_data[is_infected]["day"]

    inf_time_df = pd.DataFrame({"pid" : pid, "day" : day})

    is_rec = disease_data["state"] == "R"
    pid = disease_data[is_rec]["pid"]
    day = disease_data[is_rec]["day"]

    rec_time_df = pd.DataFrame({"pid" : pid, "day" : day})

    def lookup_rec(row):
        pid_subset = rec_time_df[(rec_time_df["pid"] == row["pid"]) & (rec_time_df["day"] >= row["day"])]
        return pid_subset["day"].min()

    recovery_times = inf_time_df.apply(lookup_rec, axis=1)
    si_table = inf_time_df
    si_table.rename({"day" : "infected"}, axis=1, inplace=True)
    si_table["recovery"] = recovery_times 

    si_table.set_index(["pid", "infected"], verify_integrity=True, inplace=True)

    return si_table
def comp_node_buckets(subgraph, ego_pid, si_window, steps = [1,2,3,4]):
    # si_window is the subgraph infection table set to the
    # appropriate time window

    vid = subgraph.vs.find(name=ego_pid)
    paths = subgraph.get_all_shortest_paths(vid)
    
    boundary_node_map = {s : set() for s in steps}
    
    for path in paths:
        if len(path) - 1 in boundary_node_map:
            path_pids = [subgraph.vs["name"][v] for v in path]
            boundary_node_map[len(path)-1].update(path_pids)
            
    output = np.zeros((len(steps),))
    
    for i,s in enumerate(steps):
        boundary_inf = si_window.index.get_level_values("pid").isin(boundary_node_map[s])
        output[i] = boundary_inf.sum()
        
    return output

def get_subgraph_inf_neighbor_data(subgraph, ego_pid, si_table, delta, cutoff, past_window):
    
    subgraph_pids = subgraph.vs["name"]
    si_subgraph = si_table[si_table.index.get_level_values("pid").isin(subgraph_pids)]
    
    if len(si_subgraph) == 0:
        return None, None
    
    ego_vid = subgraph.vs.find(name=ego_pid)
    neighbors = subgraph.neighbors(ego_vid)
    #neighbor_pid = [subgraph_pids[i] for i in neighbors]
    #si_neighbors = si_subgraph[si_subgraph.index.get_level_values("pid").isin(neighbor_pid)]
    valid_days = si_subgraph.index.get_level_values("infected").unique()
    if cutoff: 
        valid_days = valid_days[valid_days >= cutoff].unique()
    
    outputs = np.zeros((len(valid_days), 6))
    labels = np.zeros((outputs.shape[0],))
    
    i = 0
    
    for day in valid_days:
     
        min_day = max(0, day - past_window)
        
        back_subgraph = si_subgraph[si_subgraph.index.get_level_values("infected").to_series().between(min_day, day).values]
        #back_neighbors = si_neighbors[si_neighbors.index.get_level_values("infected").to_series().between(min_day, day).values]
        past_inf = si_subgraph[(si_subgraph.index.get_level_values("pid") == ego_pid) & \
                               (si_subgraph.index.get_level_values("infected") <= day)]

        subgraph_t = len(back_subgraph.index.get_level_values("pid").unique())
        #neighbor_t = len(back_neighbors.index.get_level_values("pid").unique())
        
        state_t_d = si_subgraph[si_subgraph.index.get_level_values("infected").to_series().between(day, day+delta, inclusive="right").values]
        
        outputs[i, 0] = subgraph_t
        #outputs[i, 1] = neighbor_t
        outputs[i, 1:3] = comp_node_buckets(subgraph, ego_pid, back_subgraph, [1,2])
        outputs[i, 3] = len(past_inf) > 0
        labels[i] = ego_pid in state_t_d.index.get_level_values("pid")
        
        for j,step in enumerate([1,2]):
            outputs[i,4+j]  = get_subgraph_inf_weight(back_subgraph, subgraph, ego_pid, step)
        i += 1
        
    return outputs, labels

def make_data(pid, graph, disease, pop, past_window, cutoff, delta=7):
    """
    do age, number of infected neighbors
    """
    vid = graph.vs.find(name=pid)
    neighbors = graph.neighborhood(vid, order=2)
    subgraph = graph.induced_subgraph(neighbors+[vid])

    outputs, labels = get_subgraph_inf_neighbor_data(subgraph, pid, disease, delta, cutoff, past_window)
    if outputs is None:
            return pd.DataFrame({})
    out_df = pd.DataFrame({"subgraph_inf_t" : outputs[:,0],
                           "inf_1" : outputs[:,1],
                           "inf_2" : outputs[:,2],
                           "was_inf" : outputs[:,3],
                           "inf_weight_1" : outputs[:,4],
                           "inf_weight_2" : outputs[:,5],
                           "s_t+d" : labels, 
                           "age" : pop.loc[pid, "age"], 
                           "pid" : pid})
            
    return out_df

if __name__ == "__main__":

    parser = ArgumentParser()
    #parser.add_argument("dir_name", help="directory with random walks to target")
    parser.add_argument("graph_file", help="pickled graph object")
    parser.add_argument("disease_file", help="disease data")
    parser.add_argument("pop_file", help="population characteristics file")
    parser.add_argument("out_file", help="output file for data")
    parser.add_argument("--min-date", default=0, type=int, help="Min date to generate data from (used for making evaluation set)")
    parser.add_argument("--is-eval", action="store_true",default=False, help="exclude positive instances (assume input file is training)") 
    parser.add_argument("--past-window", default=3, help="How much history to consider for each data point")
    parser.add_argument("--pid_partition", type=int, help="index of pids partition")
    parser.add_argument("--n_jobs", type=int, help="total number of jobs")

    args = parser.parse_args()

    pop = pd.read_csv(args.pop_file)
    disease_data = pd.read_csv(args.disease_file)
    si_table = make_si_table(disease_data)
    pop.set_index("pid", inplace=True)

    graph = Graph.Read_Pickle(args.graph_file)
    data = []

    disease = si_table[si_table.index.get_level_values("infected") >= (args.min_date - args.past_window)]
    vertex_names = list(graph.vs["name"])
    vertex_names.sort()

    part_size = ceil(len(vertex_names)/args.n_jobs)
    pid_part = vertex_names[args.pid_partition*part_size:(args.pid_partition+1)*part_size]

    print(f"Processing {args.pid_partition} partition of len {len(pid_part)}")

    for pid in pid_part:

        pid_data = make_data(pid, graph, disease, pop, args.past_window, args.min_date)

        if args.is_eval and len(pid_data) > 0:
            data.append(pid_data[pid_data["s_t+d"] != 1.0])
        else:
            data.append(pid_data)

    train = pd.concat(data)
    train.to_csv(args.out_file)
