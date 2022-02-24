# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:52:17 2020

@author: mmousa4
"""
##referenec: https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/09-fcma.html#connectome
##reference: https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/09-fcma.html#connectome



import numpy as np
import pandas as pd
import networkx as nx

def CreateGraph(edges):
#    edges = correlations_ICA[i,:,:]
    G = nx.Graph()
    #, coords_connectome
    epoch_corr = edges
    # What is the (absolute) correlation threshold
    threshold = 0.129
    
    nodelist = []
    edgelist = []
    
    #Normalized [0,1]
    edges = (edges-np.min(edges))/(np.max(edges)-np.min(edges))
    
    
    for row_counter in range(epoch_corr.shape[0]):
        nodelist.append(str(row_counter))  # Set up the node names
        
        for col_counter in range(epoch_corr.shape[1]):
            
            # Determine whether to include the edge based on whether it exceeds the threshold
            if abs(epoch_corr[row_counter, col_counter]) > threshold:
                # Add a tuple specifying the voxel pairs being compared and the weight of the edge
                edgelist.append((str(row_counter), str(col_counter), {'weight': epoch_corr[row_counter, col_counter]}))#1}))#
                #weight = 1: for binary Graph , binarized, undirected graph
    # Create the nodes in the graph
    G.add_nodes_from(nodelist)
    
    # Add the edges
    G.add_edges_from(edgelist)
    
    #remove self loops 
    G.remove_edges_from(nx.selfloop_edges(G))
#    print (edgelist)
#    nx.draw(G)
    return G



#%%
#reference: https://whitakerlab.github.io/scona/_modules/scona/graph_measures.html
def calc_nodal_partition(G):
    '''
    Calculate a nodal partition of G using the louvain algorithm as
    iBrainNetworkommunity.best_partition`

    Note that this is a time intensive process and it is also
    non-deterministic, so for consistency and speed it's best
    to hold on to your partition.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        A binary graph

    Returns
    -------
    (dict, dict)
        Two dictionaries represent the resulting nodal partition of G. The
        first maps nodes to modules and the second maps modules to nodes.
    '''
    import community
    # Make sure the edges are binarized
    for u, v, d in G.edges(data=True):
        if d.get('weight', 1) != 1:
            raise ValueError("G should be a binary graph")
    # Now calculate the best partition
    nodal_partition = community.best_partition(G)

    # Reverse the dictionary to record a list of nodes per module, rather than
    # module per node
    module_partition = {}
    for n, m in nodal_partition.items():
        try:
            module_partition[m].append(n)
        except KeyError:
            module_partition[m] = [n]

    return nodal_partition, module_partition


    
def participation_coefficient(graph, partition):
    """
    Computes the participation coefficient for each node.

    ------
    Inputs
    ------
    graph = networkx graph
    partition = modularity partition of graph

    ------
    Output
    ------
    List of the participation coefficient for each node.

    """
#    graph, partition = graph,module_partition
    pc_dict = {}
    all_nodes = set(graph.nodes())
    paths = dict(nx.shortest_path_length(G=graph))
    for m in partition.keys():
        mod_list = set(partition[m])
        between_mod_list = list(set.difference(all_nodes, mod_list))
        for source in mod_list:
            degree = float(nx.degree(G=graph, nbunch=source))
            count = 0
            for target in between_mod_list:
                if  source in paths and target in paths[source] and paths[source][target] == 1:
                    count += 1
            bm_degree = count
            pc = 1 - (bm_degree / degree) ** 2 if degree !=0 else 0
            pc_dict[source] = pc
    return pc_dict

#%%
#10 local and 13 global graph measures were
#calculated based on rs-fMRI adjacency matrix. The local graph
#measures were betweenness centrality, clustering coefficient,
#characteristic path, community structure Newman (CSN), community
#structure Louvain (CSL), eigenvector centrality, rich club
#coefficient, sub graph centrality,  eccentricity,and participation coefficient?
#(45). 
#The average shortest path length between all pairs of nodes in thenetwork is known as thecharacteristic path lengthof the network(e.g.,Watts and Strogatz, 1998) 
#The node eccentricity is the maximal shortest path length between a node and any other node
    
from networkx import algorithms
from community import community_louvain
from community import best_partition #conda install -c conda-forge python-louvain (amaconda prompt)
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.distance_measures import eccentricity

#reference: 2017_wang et al _Depression Disorder Classification of fMRI Data Using Sparse.pdf
# In this paper, eight graph-based features are computed from the following four aspects:
# functional segregation (Clustering coefficient,Local efficiency), 
# functional integration (Characteristic path length, global efficiency ), 
# nodal centrality (Degree and betweenness centrality), and 
# network resilience
def feature_vector2(G):
    # 4 local + 3 Global
#   G=graph
    featureVector=[]
    
    #functional segregation (Clustering coefficient,Local efficiency)
    #-----------------------------------
    #Clustering coefficient
    glm_l_clusteringCoefficient= nx.clustering(G)
    
    #Local efficiency
    glm_g_local_efficiency = nx.algorithms.efficiency_measures.local_efficiency(G)
    
    #functional integration (Characteristic path length, global efficiency )
    #----------------------------------
    #Characteristic path length
    if nx.is_connected(G):
        glm_g_characteristicPath= nx.average_shortest_path_length(G)
    else:
        glm_g_characteristicPath = 0
    
    #global efficiency
    glm_g_global_efficiency = nx.global_efficiency(G)
    
   
    #Nodal centrality (Degree and betweenness centrality)
    #----------------------------------
    #Degree
    glm_l_degree = G.degree
    degreelist = [d for n, d in G.degree()]
    glm_l_degree = dict(zip(range(0,len(degreelist)), degreelist))     
    #Betweenness centrality
    glm_l_betweennessCentrality=nx.betweenness_centrality(G,normalized=True)
    
    #Participation coefficient
    nodal_partition, module_partition= calc_nodal_partition(G)
    glm_l_pp =participation_coefficient (G,module_partition)
    #Network Resilience
    #-----------------------------------
    glm_l_average_neighbor_degree = nx.average_neighbor_degree(G)
    
    #Once we have obtained all the eight graph-based features, we concatenate them to construct the final feature vectors.
    #Specifically, for each subject, the feature vector has a size of 698, which consists of 116 ∗ 6 local measures and 2 global ones.
    featureVector=list(glm_l_clusteringCoefficient.values())
#    featureVector = featureVector + list(glm_l_local_efficiency)
    featureVector = featureVector + list(glm_l_degree.values())
    featureVector = featureVector + list(glm_l_betweennessCentrality.values())
    featureVector = featureVector + list(glm_l_average_neighbor_degree.values())
    featureVector = featureVector + list(glm_l_pp.values())
    
    featureVector.append(glm_g_characteristicPath if nx.is_connected(G) else 0)
    featureVector.append(glm_g_global_efficiency)
    featureVector.append(glm_g_local_efficiency)
    
#    fv_columns = []
    fv_columns = [str(x) + 'clusteringCoefficient' for x in list(glm_l_clusteringCoefficient.keys())] 
#    fv_columns = fv_columns + [str(x) + 'local_efficiency' for x in list(glm_l_local_efficiency.keys())]  
    fv_columns = fv_columns + [str(x) + 'nodeDegree' for x in list(glm_l_degree.keys())]
    fv_columns=  fv_columns +[str(x) + 'betweennessCentrality' for x in list(glm_l_betweennessCentrality.keys())] 
    fv_columns = fv_columns + [str(x) + 'average_neighbor_degree' for x in list(glm_l_average_neighbor_degree.keys())]
    fv_columns = fv_columns + [str(x) + 'participation_coefficient' for x in list(glm_l_pp.keys())]
    
    fv_columns = fv_columns + ['characteristicPath','global_efficiency','local_efficiency']
    print("featureVector.length:",featureVector.count)
    print(fv_columns)
    return featureVector , fv_columns
    
def feature_vector(G):
    #6 local and 8 global graph measures were computed that resultedin 913 features
#    G=graph
    featureVector=[]
    #Local measures
    #--------------
    glm_l_betweennessCentrality=nx.betweenness_centrality(G) #1.betweenness centrality (defined as the fraction of all shortest paths in the network that pass through a given node.)
    glm_l_clusteringCoefficient= nx.clustering(G)#2.clustering coefficient(cc)
#    glm_l_characteristicPath = dict(nx.all_pairs_shortest_path_length(G)) #3.characteristic path
    #3. The average shortest path length between all pairs of nodes in the network is known as the characteristic path length of the network
    glm_l_csn = nx.algorithms.community.girvan_newman(G) #4.community structure Newman (CSN)
    glm_l_csn = tuple(sorted(c) for c in next(glm_l_csn))
    #glm_l_csl =community_louvain.best_partition(G) #5.community structure Louvain (CSL) # use community of 'python-louvain' vs 'networkx.algorithms.community' 
    glm_l_evcentrality = nx.eigenvector_centrality(G) #6.eigenvector centrality
    glm_l_richClubCoef = nx.rich_club_coefficient(G, normalized=False)#7.rich_club_coefficient is not implemented for graphs with self loops.
    glm_l_subGraphCentrality = nx.algorithms.centrality.subgraph_centrality(G) #8.sub graph centrality
    if nx.is_connected(G):
        glm_l_eccentricity = eccentricity(G) #9.eccentricity
        glm_g_characteristicPath= float(nx.average_shortest_path_length(G))#3.characteristic path
        glm_g_diameter = nx.diameter(G)#9.graph diameter
        glm_g_smallworld_sigma = nx.algorithms.smallworld.sigma(G)#11.small-worldness

    
    #The global graph measures were assortativity, clustering
    #coefficient, characteristic path, community structure Newman
    #output, community structure Louvain output, cost efficiency
    #(two measures), density, efficiency, graph radius, graph diameter,
    #transitivity, and small-worldness (45).
    #The global efficiency is the average inverse shortest path length in the network
    #The global cost efﬁciency is then deﬁned as the global efﬁciency at a given cost minus the cost,i.e.,(E	C),which will typically have a maximum value max(E	C)0,atsomecostCmax,foraneconomicalsmall-worldnetwork.Likewise,the regionalcostefﬁciencywascalculatedasthemaximumofthefunction(E(i)	k),wherekisthedegreeornumberofedgesconnectingtheith
    #Global measures
    #--------------
#    glm_g_degree_assortativity_coef = nx.algorithms.assortativity.degree_assortativity_coefficient(G) # 1.assortativity
    glm_g_clusteringCoefficient=nx.average_clustering(G) #2.Global Clustering Coefficient (CC)
        #4.community structure Newman output
    #5.community structure Louvain output
#    glm_g_globalCostEfficiency = glm_g_global_efficiency - #6.cost efficiency(two measures)
    #GCE=E - PSW, Where Ei is the efficiency of node i, N is the set of all nodes in the network, n is the number of nodes and di j is the shortest path length (distance) between nodes i and j
    glm_g_density = nx.density(G)#7.density
    glm_g_global_efficiency = nx.global_efficiency(G)#8.efficiency 
    
    glm_g_transitivity= nx.transitivity(G)#10.transitivity
    glm_g_radius = nx.radius(G)#12.graph radius

    
    featureVector=list(glm_l_betweennessCentrality.values())
    featureVector = featureVector + list(glm_l_clusteringCoefficient.values())
#    featureVector = featureVector + list(glm_l_characteristicPath.values())
    featureVector = featureVector + list(glm_l_evcentrality.values())
    featureVector = featureVector + list(glm_l_richClubCoef.values())
    featureVector = featureVector + list(glm_l_subGraphCentrality.values())
    featureVector = featureVector + list(glm_l_eccentricity.values())
    
#    featureVector.append(glm_g_degree_assortativity_coef)
    featureVector.append(glm_g_clusteringCoefficient)
    featureVector.append(glm_g_characteristicPath)
    featureVector.append(glm_g_density)
    featureVector.append(glm_g_global_efficiency)
    featureVector.append(glm_g_diameter)
    featureVector.append(glm_g_transitivity)
    featureVector.append(glm_g_smallworld_sigma)
    featureVector.append(glm_g_radius)
    
    
#    fv_columns = []
    fv_columns=  [str(x) + 'betweennessCentrality' for x in list(glm_l_betweennessCentrality.keys())] 
    fv_columns = fv_columns + [str(x) + 'clusteringCoefficient' for x in list(glm_l_clusteringCoefficient.keys())] 
#    fv_columns = fv_columns + list(glm_l_characteristicPath.values())
    fv_columns = fv_columns + [str(x) + 'evcentrality' for x in list(glm_l_evcentrality.keys())]  
    fv_columns = fv_columns + [str(x) + 'richClubCoef' for x in list(glm_l_richClubCoef.keys())]   
    fv_columns = fv_columns + [str(x) + 'subGraphCentrality' for x in list(glm_l_subGraphCentrality.keys())] 
    fv_columns = fv_columns + [str(x) + 'eccentricity' for x in list(glm_l_eccentricity.keys())] 
    fv_columns = fv_columns + ['clusteringCoefficient','characteristicPath','density','global_efficiency','diameter','transitivity','smallworld','radius']
    print("featureVector.length:",featureVector.count)
    print(fv_columns)
    return featureVector , fv_columns#'degree_assortativity_coef',
#%%
import os 

#np.load.__defaults__=(None, False, True, 'ASCII')
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#np.load = lambda *a,**k: np_load_old(*a,allow_pickle=True)

#os.chdir('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\NewResults')
#train_feature= np.load('ExpandfMRI_UNET2D_VGG162D.npy')
#final_labels_train_3d = np.load('final_labels_train_3d.npy')
#
#corr_den= np.load('corr_den.npy')
#from nilearn.connectome import GroupSparseCovarianceCV,sym_matrix_to_vec,ConnectivityMeasure
#connectivities = sym_matrix_to_vec(corr_den, discard_diagonal=True)
#final_labels = np.load('final_labels.npy')

#ts_canica = np.load('ts_canica.npy')
#covariences= np.load('canica_GL_covariences.npy')
prefix = 'aal_corr_den'
from nilearn.connectome import GroupSparseCovarianceCV,sym_matrix_to_vec,ConnectivityMeasure
correlations= np.load(prefix+'.npy')

#connectivities = sym_matrix_to_vec(covariences, discard_diagonal=True)
#final_labels = np.load('final_labels.npy')
#ind = np.load('indx_rus_9_646431.npy')
#connectivities=connectivities[ind]
##final_labels=final_labels[ind]
#covariences = covariences[ind]
#%%
#correlations=np.array(cov,dtype='float16')   
#correlations= corr_den
#correlations=covariences

fv_list=[]
#nNodes = []
df = pd.DataFrame()

for i in range (0,correlations.shape[0]):
    
    graph = CreateGraph(correlations[i,:,:])
#    print(graph)
    fv,column = feature_vector2(graph)
    df1 = pd.DataFrame([fv], columns =list(column))     
    df = df.append(df1, sort = False)
df = df.reset_index(drop=True)
graphFeatures = df.values

#df.to_csv('graphdf_canica.csv')
#np.save('graphFeatures_canica.npy',np.array(graphFeatures,dtype='float16'))
#np.save('graphFeatures_SpectrumDensity.npy',np.array(graphFeatures,dtype='float16'))
np.save(prefix+'graphFeatures.npy',np.array(graphFeatures,dtype='float16'))

