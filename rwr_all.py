"""
Author: Galen Harrison (gh7vp@virginia.edu)
Date: 2022-08-11

rwr_all is performs and stores random walks with restart (RWR) starting from a 
subset of vertices in a graph.

The graph file should be a pickled igraph Graph object, produced e.g. by
igraph's graph.write_pickle method. The graph is assumed to have vertices
with attribute "name" and edges with attribute "duration"

The script divides the graph's vertices into n_jobs pieces, and produces
random walks for the portion indexed by pid_partition.

The output is a directory specified by --output. The script will generate
a set of pickled dictionary files in this directory each with size dict_size

This script may be stopped and restarted, and if the output directory is the
same between stops and restarts, will skip already-processed vertices within
the partition
"""

import os

from math import ceil
from pickle import dump
from argparse import ArgumentParser
from random import random

import numpy as np
from igraph import Graph

def rand_walk_igraph(graph, start_node, size, restart_prob, wname="weight"):
    """
    Do an edge-weighted random walk with restart starting from start_node
    on graph.

    graph - an igraph graph with vertex attributes "name" and edge attributes wname
    start_node - a vertex in graph (note that this *should not* be the vertex name)
    size - the maximum size of the random walk
    restart_prob - the probability of returning to start_node at each step
    wname (optional) - the name of the edge attribute to weight

    returns - subgraph (an igraph graph produced through the random walk)
    """
    nodes = set([start_node])
    current = start_node

    # To avoid a situation where we end up in a loop, we limit the number
    # of steps we can take without adding a new node

    max_step = 100
    step = 0

    while len(nodes) < size:
        curr_size = len(nodes)
        if random() < restart_prob or len(graph.neighbors(current)) == 0:
            current = start_node
        else:
            poss_edges = graph.incident(current)
            poss_weights = np.array([graph.es[n][wname] for n in poss_edges])
            if len(poss_edges) == 0:
                print(f"There are no incident edges for {current}, but neighbors {graph.neighbors(current)}")
            if poss_weights.sum() == 0:
                print(f"Edge weights are 0 for {current}")
                current = start_node
            else:
                poss_weights = np.nan_to_num(poss_weights)
                new_edge = np.random.choice(poss_edges, p = poss_weights/poss_weights.sum())

                if graph.es[new_edge].source == current:
                    current = graph.es[new_edge].target
                else:
                    current = graph.es[new_edge].source 
            nodes.add(current)
        if len(nodes) == curr_size:
            step += 1
        else:
            step = 0
        if step == max_step:
            print(f"rwr - too many steps for graph {start_node} {len(nodes)}") 
            return graph.induced_subgraph(list(nodes))
    print(f"Finished node size {start_node} {len(nodes)}")
    return graph.induced_subgraph(list(nodes))
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("graph_file", help="pickled igraph graph")
    parser.add_argument("--pid_partition", type=int, help="index of pids partition")
    parser.add_argument("--n_jobs", type=int, help="total number of jobs")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--dict_size", type=int, default=1000, help="size of dictionaries to write out")
    parser.add_argument("--rwr-size", type=int, default=50, help="maximum size of sampled random walk")
    parser.add_argument("--restart-prob", type=float, default=0.8, help="restart probability")
    args = parser.parse_args()

    graph = Graph.Read_Pickle(args.graph_file)
    vertex_names = list(graph.vs["name"])
    vertex_names.sort()

    print(f"There are {len(vertex_names)} vertices")

    part_size = ceil(len(vertex_names)/args.n_jobs)
    start = args.pid_partition*part_size
    end = start + part_size

    print(f"Doing partition from {start} to {end} ({vertex_names[start]} to {vertex_names[min(end-1, len(vertex_names)-1)]}")
    pid_part = vertex_names[args.pid_partition*part_size:(args.pid_partition+1)*part_size]
    graph_dict = {}
    n_dict = 0

    for pid in pid_part:
        # We keep track of already-processed sections of pid_part 
        # through the filename. If the filename for this chunk already exists,
        # we skip until we find a pid section we have not seen yet.
        filename = os.path.join(args.output, f"subgraph_{args.pid_partition}_{n_dict}.rwr_table")
        if os.path.exists(filename):
            graph_dict[pid] = None
            if len(graph_dict) == args.dict_size:
                graph_dict = {}
                n_dict += 1
            continue 
        else:
            node = graph.vs.find(name=pid) # name
            subgraph = rand_walk_igraph(graph, node, args.rwr_size, args.restart_prob, wname="duration")
            graph_dict[pid] = subgraph

        # If we have reached the dictionary size, write out the dict to file
        # and start new dictionary object
        if len(graph_dict) == args.dict_size: 
            with open(filename, "wb") as outfile:
                filter_dict = {pid : graph for pid,graph in graph_dict.items() if graph}
                dump(filter_dict, outfile)
            graph_dict = {}
            n_dict += 1
