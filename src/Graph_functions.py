# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:17:31 2020

@author: 108431
"""

import networkx as nx
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt

def set_node_community(G, communities):
        '''Add community to node attributes'''
        for com, nodes  in communities.items():
            for node in nodes:
                # Add 1 to save 0 for external edges
                G.nodes[node]['community'] = com

def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0
    
def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a community.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)   

def hierarchicalClustering (G, cluster_size_threshold = 5):
    # Create partitions with hierarchical clustering
    path_length = dict(nx.shortest_path_length(G, weight='weight'))
    distances = np.zeros((len(G),len(G)))
    #plt.imshow(distances, cmap='hot', interpolation='nearest')
    for u,p in path_length.items():
        for v,d in p.items():
            distances[u][v]=d
    # Create hierarchical cluster
    Y = distance.squareform(distances)
    Z = hierarchy.single(Y)      
    dn = hierarchy.dendrogram(Z)

    hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y', orientation='top')
    dn2 = hierarchy.dendrogram(Z, ax=axes[1], above_threshold_color='#bcbddc', orientation='right')
    hierarchy.set_link_color_palette(None)  # reset to default after use
    plt.show()
    
    cophenetic_dist = hierarchy.cophenet(Z)
    # at these distances the clusters or inital points were joined together (and sorted)
    distance_decissions=  sorted(list(set(hierarchy.cophenet(Z))))
    
    previousPartition = list(G.nodes)
    for cophenetic_dist in distance_decissions:
        partition = hierarchy.fcluster(Z, cophenetic_dist, criterion='distance')
        clustersSize = Counter(partition)
        if (len(np.where(np.array(list(clustersSize.values())) > cluster_size_threshold)[0])>0):
            partition = previousPartition
            break
        else:
            previousPartition = partition
    
    partition = dict(list(zip(list(G.nodes),partition)))
    
    for node in G.nodes:
        G.nodes[node]['community'] = partition[node]
    return G, partition     
    
