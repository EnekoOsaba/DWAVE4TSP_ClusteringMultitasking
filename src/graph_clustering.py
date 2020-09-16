# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:14:08 2020

@author: 108431
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from networkx.algorithms.community import modularity
import Graph_functions, CommunityPainting
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random


distancesDict = {0:'silhouette', 1:'modularity', 2:'davies_bouldin_score', 3:'calinski_harabasz_score'}
metricsToApply = sorted([0,2,3])

subPopulationSize = 20
generations = 100

knowledgeTransferRate = 0.2 # [0,1]
knowledgeTransferSizeProportion = 0.10 # [0,1]

populationSize = subPopulationSize * len(metricsToApply)
knowledgeTransferSize = int(knowledgeTransferSizeProportion * subPopulationSize)

#print('{} individuals will be transferred'.format(knowledgeTransferSize))


# swapping operator
def swapping(population, prob = 0.10):
    newPopulation = population.copy()
    for individual in newPopulation:
        swapTimes = int(len(individual)*prob)
        #print('Number of changes in the individual', swapTimes)
        #print('individual', individual)
        for swap in range(swapTimes):
            ## Two exclusive indices, False for replacement so they are exclusive
            x1, x2 = np.random.choice(range(len(individual)), 2, False)
            #print('swapping between', x1, x2, ' changing values', individual[x2], individual[x1])
            ## Swap the genes at the indices
            individual[x1], individual[x2] = individual[x2], individual[x1]
            #print ('new values x1,x2:', individual[x1], individual[x2])
            #print('new individual', individual)
    return newPopulation

# insertion operator
def insertion(population, prob = 0.10):
    newPopulation = population.copy()
    for individual in newPopulation:
        insertionTimes = int(len(individual)*prob)
        #print('individual', individual)
        for insertion in range(insertionTimes):
            whereToInsert, whatToInsert = sorted(np.random.choice(range(len(individual)), 2, False))
            #print('insertion en position', whereToInsert, 'with value from position', whatToInsert)
            individual = np.insert(individual, whereToInsert, individual[whatToInsert])
            individual = np.delete(individual,whatToInsert+1)
            #print('new individual', individual)
    return newPopulation

#2-OPT operator
def two_opt(population, prob = 0.10):
    newPopulation = population.copy()
    for individual in newPopulation:
        twoOptTimes = int(len(individual)*prob)
        #print('individual', individual)
        for twoOpt in range(twoOptTimes):
            x1, x2 = sorted(np.random.choice(range(len(individual)), 2, False))
            individual[x1:x2] = individual[x1:x2][::-1]
            #print('2-opt - fragment', x1, x2, ' reversing result', individual)
    return newPopulation

# PARAMS: distanceToApply=array of integers (size=population) according to distancesDict
# distancesDict = {0:'silhouette', 1:'modularity', 2:'davies_bouldin_score', 3:'calinski_harabasz_score'}
def _fitness(distances_df, points, population, distanceToApply, subPopulationSize = subPopulationSize, best=True):       
    scores = []       
    for individualIdx in range(len(population)):
        individual = population[individualIdx]
        if distancesDict[distanceToApply]=='silhouette':
            #The best value is 1 and the worst value is -1
            scores.append(silhouette_score(distances_df.to_numpy(), individual, metric = 'precomputed'))   
        elif distancesDict[distanceToApply]=='modularity':
            # range: [−1/2,1) Best value is 1
            scores.append(modularity(nx.from_numpy_matrix(distances_df.to_numpy()), [np.where(individual==cluster)[0] for cluster in set(individual)], weight='weight'))
        # lower values indicating better clustering
        elif distancesDict[distanceToApply]=='davies_bouldin_score':
            scores.append(-davies_bouldin_score(points, individual))
        # biggest values indicating better clustering
        elif distancesDict[distanceToApply]=='calinski_harabasz_score':
            scores.append(calinski_harabasz_score(points, individual))
    if best:
        selectedScoresIdx = np.argsort(scores)[::-1][:subPopulationSize]      
    else:
        selectedScoresIdx = np.argsort(scores)[:subPopulationSize]      
    return selectedScoresIdx, np.max(scores) if best else np.min(scores)
    
def reduction(distances_df, points, wholePopulation, metricsToApply, subPopulationSize = subPopulationSize, end = False): 
    newPopulation = []    
    if end:
        bestScores = {metricIdx:0 for metricIdx in metricsToApply}
        for subPopulationIdx in range(len(metricsToApply)): # 3 subpopulations, concatenate subpopulations from pupulations and get fitness
            subPopulation = wholePopulation[subPopulationIdx*subPopulationSize:subPopulationIdx*subPopulationSize+subPopulationSize]
            fittest, bestScore = _fitness(distances_df, points, subPopulation, metricsToApply[subPopulationIdx], 1)  
            newPopulation.append(subPopulation[fittest[0]])
            bestScores[metricsToApply[subPopulationIdx]] = bestScore
    else:
        fittest, bestScore = _fitness(distances_df, points, wholePopulation, metricsToApply, subPopulationSize)  
        for item in fittest: 
            newPopulation.append(wholePopulation[item])          
    return np.array(newPopulation), bestScores if end else bestScore 

def transferringKnowledge(distances_df, points, population, metricsToApply, subPopulationSize, knowledgeTransferSize = knowledgeTransferSize):
    newPopulation = population.copy()
    fittestIdx = []
    worstIdx = []
    for subPopulationIdx in range(int(len(newPopulation)/subPopulationSize)): 
        subPopulation = newPopulation[subPopulationIdx*subPopulationSize:subPopulationIdx*subPopulationSize+subPopulationSize]
        fittest = _fitness(distances_df, points, subPopulation, metricsToApply[subPopulationIdx], knowledgeTransferSize) [0]
        fittestIdx.append([subPopulationIdx*subPopulationSize+fittestItem for fittestItem in fittest])
        worst = _fitness(distances_df, points, subPopulation, metricsToApply[subPopulationIdx], knowledgeTransferSize, best=False) [0]
        worstIdx.append([subPopulationIdx*subPopulationSize+worstItem for worstItem in worst])
    #print(fittestIdx,worstIdx)
    knowledgeTransferIdx = []
    for subPopulationIdx in range(int(len(newPopulation)/subPopulationSize)):
        # (badIndividualsToBeDestroyed, goodIndividualsToCopy)
        knowledgeTransferIdx.append((subPopulationIdx, np.random.choice([choice for choice in list(range(int(len(newPopulation)/subPopulationSize))) if choice!=subPopulationIdx])))
    for exchange in knowledgeTransferIdx:
        exchangeFittestIdx = fittestIdx[exchange[1]]
        exchangeWorstIdx = worstIdx[exchange[0]]
        #print('exchanging ', exchangeWorstIdx, 'for ', exchangeFittestIdx)
        for transfer in range(knowledgeTransferSize):
            newPopulation[exchangeWorstIdx[transfer]] = newPopulation[exchangeFittestIdx[transfer]]
    return newPopulation

def solvePartitionProblem(cluster_size,tsp_matrix,node_array):

    cluster_size_threshold = cluster_size
    
    # Problem statement
    
    numItems = len(tsp_matrix)
    points = node_array
    #G = nx.from_numpy_matrix(tsp_matrix)
    distances_df = pd.DataFrame(tsp_matrix)
     
       
    # PROBLEM CONFIG
    population = []  
    numClusters = int(numItems / cluster_size_threshold) if (numItems % cluster_size_threshold)==0 else int(numItems / cluster_size_threshold) + 1
    numItemsPerCluster_dict = {cluster: [cluster]*(int(numItems / numClusters) + 1) if cluster<= (numItems % numClusters) else [cluster]*(int(numItems / numClusters)) for cluster in range(1, numClusters + 1)}
    vocabulary = np.array([value for valueList in list(numItemsPerCluster_dict.values()) for value in valueList]) # possible values for the individuals
    
    for _ in range(populationSize): 
        random.shuffle(vocabulary) 
        population.append(vocabulary.copy())
    population = np.array(population)
    
    # generations
    scoreEvolution = {metric:[] for metric in metricsToApply}
    for _ in range(generations):  
        if _%10==0:
            print('#Generation',_)
        newPopulation = []
        for subPopulationIdx in range(len(metricsToApply)): # 3 subpopulations, concatenate subpopulations from pupulations and get fitness
            subPopulation = np.array([individual for individual in population[subPopulationIdx*subPopulationSize:subPopulationIdx*subPopulationSize+subPopulationSize]])
            newSubpopulation_twoopt = two_opt(subPopulation, prob=1/numItems) 
            newSubpopulation_insertion = insertion(subPopulation, prob=1/numItems)
            newSubpopulation_swap = swapping(subPopulation, prob=1/numItems)
            newSubPopulation, bestScore = reduction(distances_df, points, np.concatenate((subPopulation, newSubpopulation_twoopt, newSubpopulation_insertion, newSubpopulation_swap)), metricsToApply[subPopulationIdx], subPopulationSize) 
            scoreEvolution[metricsToApply[subPopulationIdx]].append(bestScore)
            newPopulation.append(newSubPopulation)
        population = np.concatenate(newPopulation)
        if _>0 and _%int(generations*knowledgeTransferRate)==0:
            print('Transferring knowledge in generation {}'.format(_))
            population = transferringKnowledge(distances_df, points, population, metricsToApply, subPopulationSize, knowledgeTransferSize) 
    
    winners, bestScores = reduction(distances_df, points, population, metricsToApply, subPopulationSize, True) 
    winners = {metricsToApply[metricRelativeIdx]:winners[metricRelativeIdx] for metricRelativeIdx in range(len(metricsToApply))}
    print('Winners {0} with scores {1}'.format(winners,bestScores))
    
#    for metricIdx in metricsToApply:
#        metricName = distancesDict[metricIdx]
#        
#        partition = dict(zip(list(G.nodes),list(winners[metricIdx])))
#        for node in G.nodes:
#            G.nodes[node]['community'] = partition[node]
#        
#        #plot scoreEvolution    
#        fig, ax = plt.subplots()
#        plt.plot(scoreEvolution[metricIdx])
#        plt.title("Score Evolution:{}".format(metricName)) 
#        
#        # Communities Visualization
#        node_color_dict = {key: Graph_functions.get_color(G.nodes[key]['community']) for key in G.nodes}
#        g_pos = CommunityPainting.community_layout(G, partition)
#        
#        fig, ax = plt.subplots()
#        nx.draw_networkx(
#            G,
#            pos = g_pos,
#            node_list= G.nodes(),
#            node_color = list(node_color_dict.values()),
#            alpha=0.8,
#            edgelist = G.edges,
#            edge_color = 'silver')
#        plt.title("Communities:{}".format(metricName))   

    return winners