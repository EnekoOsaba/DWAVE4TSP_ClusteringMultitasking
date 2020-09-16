# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:29:31 2020

@author: 108431
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np, copy
import pandas as pd


def getLinkToFollowingCluster (veryInitialEdge, linkedClustersIdx, edgeToStartFrom, clustersTSP_nodes, distances_df, clusters_dict, cyclesList, currentRoute):    
    
#    print('******************************')
    # print('linkedClustersIdx', linkedClustersIdx)   
  
    # internal link in cluster
    cluster0Node0= edgeToStartFrom[0]
    cluster0Node1= edgeToStartFrom[1]
    clusterLinked1Idx = clusters_dict[cluster0Node1]
    clusterLinked1 = clustersTSP_nodes[clusterLinked1Idx]
    clusterLinked1_nodes = list(set([elem for tup in clusterLinked1 for elem in tup]))    
    linkedClustersIdx.append(clusterLinked1Idx)   
    
    nodesPreviousClusters = [node for node in range(len(distances_df)) if (node in [node for node, cluster in clusters_dict.items() if cluster in linkedClustersIdx])]  
    # print('linkedClustersIdx after append',linkedClustersIdx)
    # print('nodes to discard', nodesPreviousClusters)
          
    potentialInternalEdgesCluster = [(cluster0Node1,edge[0] if edge[0]!=cluster0Node1 else edge[1]) for edge in clusterLinked1 if cluster0Node1 in edge]
    #print('Potential edges in other cluster',potentialInternalEdgesCluster)
    for potentialEdgeCluster in potentialInternalEdgesCluster: 
#        print('edgeToStartFrom', edgeToStartFrom)
#        print('route from potential edge',potentialEdgeCluster)
        currentRoute.append(potentialEdgeCluster)
        potentialEdgeInClusterNode0 = potentialEdgeCluster[0] #es minNode (viene del cluster previo)
        potentialEdgeInClusterNode1 = potentialEdgeCluster[1]
        # link to following cluster
        if len(nodesPreviousClusters)==len(distances_df): # no cluster to link to! go back to cluster 0!:
            # link to first cluster (end of cycle)
            minDist = distances_df[potentialEdgeInClusterNode1][veryInitialEdge[1]]           
            clusterX_0X_1Edge = (potentialEdgeInClusterNode1,veryInitialEdge[1])
            currentRoute.append(clusterX_0X_1Edge)
            cyclesList.append(copy.deepcopy(currentRoute))
#            print('clusterX_0X_1Edge',clusterX_0X_1Edge)
#            print(currentRoute)
#            print('end of route')
        else:
            minDist = distances_df[potentialEdgeInClusterNode1][distances_df.columns.difference(nodesPreviousClusters)].min()
            minNode = distances_df[potentialEdgeInClusterNode1][distances_df.columns.difference(nodesPreviousClusters)].idxmin()
            clusterX_0X_1Edge = (potentialEdgeInClusterNode1, minNode)
            currentRoute.append(clusterX_0X_1Edge)
#            print('clusterX_0X_1Edge',clusterX_0X_1Edge)
#            print('currentRoute', currentRoute)
            getLinkToFollowingCluster(veryInitialEdge, linkedClustersIdx, clusterX_0X_1Edge, clustersTSP_nodes, distances_df, clusters_dict, cyclesList, currentRoute)
        currentRoute = [currentRoute[edgeIdx] for edgeIdx in range(currentRoute.index(edgeToStartFrom)+1)]
        

def findCycle (startingCluster, clustersTSP_nodes, distances_df, clusters_dict):
    currentRoute = []
    cyclesList = []
    cluster0Idx = startingCluster
    cluster0 = clustersTSP_nodes[cluster0Idx]
    cluster0_nodes = list(set([elem for tup in cluster0 for elem in tup]))
    for edge in cluster0:
        edgeVariations = [edge, (edge[1], edge[0])]
        for edgeVariation in edgeVariations: 
            currentRoute = []
            currentRoute.append(edgeVariation)
            #print('edgeVariation',edgeVariation)
            # link to first cluster
            cluster0Node0= edgeVariation[0]
            cluster0Node1= edgeVariation[1]
            # conseguir distancia más corta al resto de elementos de otros clusters
            minDist = distances_df[cluster0Node0][distances_df.columns.difference(cluster0_nodes)].min()
            minNode = distances_df[cluster0Node0][distances_df.columns.difference(cluster0_nodes)].idxmin()
            cluster01Edge = (cluster0Node0, minNode) # edge al cluster cluster1
            currentRoute.append(cluster01Edge)
            #print(currentRoute)           
            linkedClustersIdx = [cluster0Idx]
            
            getLinkToFollowingCluster(edgeVariation, linkedClustersIdx, cluster01Edge, clustersTSP_nodes, distances_df, clusters_dict, cyclesList, currentRoute)
    return cyclesList


def composeCycle (clustersTSP_nodes, distances_df, clusters_dict):
    totalFinalCycleList = []
    winnerMinDist = np.inf
    for clusterIdx in range(len(clustersTSP_nodes)):
        totalFinalCycleList.extend(findCycle(clusterIdx, clustersTSP_nodes, distances_df, clusters_dict))
    
    distDict = {}
    for cycle in totalFinalCycleList:
        # [(9, 8), (9, 1), (1, 2), (2, 11), (11, 12), (12, 8)] --> sum internal cycle (menos los impares que son aristas dentro del cluster y que son las que realmente se rompen)
        brokenEdges = cycle[0::2]
        newEdges = cycle[1::2]  
        completeTSP = copy.deepcopy(newEdges)
        dist = np.sum([distances_df.loc[edge[0],edge[1]] for edge in newEdges])
        for cluster in clustersTSP_nodes:
            dist += np.sum([distances_df.loc[edge[0],edge[1]] for edge in cluster if edge not in brokenEdges])
            completeTSP.extend([edge for edge in cluster if edge not in brokenEdges])
        distDict[tuple(completeTSP)] = dist
    
    winner = min(distDict.items(), key=lambda it: it[1])      
    #print ('WINNER!!',  winner)
    return winner
    
    # Graph TSP 
#    distances_TSP = np.zeros(distances_df.shape)
#    for edge in winner[0]:
#        distances_TSP[edge[0],edge[1]] = distances_df.loc[edge[0],edge[1]]
#        distances_TSP[edge[1],edge[0]] = distances_df.loc[edge[0],edge[1]]
#    G_TSP = nx.from_numpy_matrix(distances_TSP)
    
    # Graph Visualization
#    fig, ax = plt.subplots()
#    plt.title(" TSP Graph")
#    pos = nx.spring_layout(G_TSP)
#    plt.axis('off')
#    nx.draw_networkx_nodes(G_TSP, pos, node_size=20, with_labels=True)
#    nx.draw_networkx_labels(G_TSP, pos)
#    nx.draw_networkx_edges(G_TSP, pos, alpha=0.4)
    # labels = nx.get_edge_attributes(G_TSP,'weight')
    # nx.draw_networkx_edge_labels(G_TSP,pos,edge_labels=labels)  

def recomposeTSPsubcycles(points,distances,list_index,list_solutions):
            
    distances_df = pd.DataFrame(distances)
    pointsIdx_cluster = []
    cluster_TSP_nodes_aux = []
    clusters_TSP_nodes = []
    
    for list_nodes in list_solutions:
        cluster_TSP_nodes_aux = []
        for i in range(len(list_nodes)):
            if i==len(list_nodes)-1:
                tuple_aux = (list_nodes[i],list_nodes[0])
            else:
                tuple_aux = (list_nodes[i],list_nodes[i+1])
            cluster_TSP_nodes_aux.append(tuple_aux)
        clusters_TSP_nodes.append(cluster_TSP_nodes_aux)
    
    clusters_dict = {node: clusterIdx for clusterIdx in range(len(list_index)) for node in list_index[clusterIdx]}
    
    return composeCycle(clusters_TSP_nodes, distances_df, clusters_dict)   
