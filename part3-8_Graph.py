# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:22:43 2020

@author: mmousa4
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter 


from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split,GroupShuffleSplit
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix, f1_score, roc_auc_score,precision_score,recall_score
from sklearn.externals.joblib import Memory
from sklearn.metrics import log_loss

import scipy
from scipy import ndarray
from scipy.ndimage import interpolation

from nilearn import image
from nilearn.image import load_img
from nilearn.decomposition import CanICA
from nilearn.decomposition import DictLearning
from nilearn.input_data import NiftiMapsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure,sym_matrix_to_vec
from nilearn.regions import RegionExtractor
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map
from nilearn.image import resample_to_img
from nilearn.image import concat_imgs


from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.layers import BatchNormalization,Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras import layers
from keras import models
from keras.applications import VGG16,inception_v3
from keras.backend import tensorflow_backend as K
from keras.callbacks import EarlyStopping

#%% Loading data
#MY Computer
#------------
di = "C:/Users/mmousa4/Documents/Greening_Data/"
smripath = di+ "participants/"
srcpath =  di +"fmriOutput/fmriprep/"#"C:\\Users\\mmousa4\\Documents\\Greening_Data\\participants\\"
labelpath="C:/Users/mmousa4/Documents/Zhachary/BIDS_Folder_Format/SubjectsInBDIFormat.xlsx"
confoundPath= di + "fmriOutput/"+ "confounds_27/"
prepath = di+ "Oversample/"
templatePath = di +'template_img_ordered.nii.gz'

#HPC
#------------
#di = "/work/mmousa4/Brain/Greening_Data/"
#smripath =  di+ "participants/"
#srcpath =  di + "Out/fmriprep/"
#labelpath="/work/mmousa4/Brain/new_data_on_400/SubjectsInBDIFormat.xlsx"
#confoundPath=di +"confounds_27/"
#prepath = di+ "Oversample/"
#templatePath = di +'template_img_ordered.nii.gz'


subjectsList= os.listdir(srcpath) 
df_label = pd.read_excel(labelpath, header =0 , index_col=2 )

func_filenames,stru_filenames=[],[]
confounds =[]
time_series_list=[]
labels=[]
for participant in subjectsList:    
    func_file= srcpath+ participant + "/func/"+ participant+ "_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    
#    strc_file = smripath + df_label.BDI_Format.get(participant,'**')  +"/V2/T1/T1.nii.gz" 
#    print (strc_file)
    if os.path.exists(func_file) and not np.isnan(df_label.loc[participant,'BDI']) and len(func_filenames) < len(subjectsList):#os.path.exists(strc_file) and 
#        stru_filenames.append (strc_file)
        
        func_filenames.append(func_file)
        labels.append(df_label.loc[participant,'BDI'])
        ##---Load existing confounds -
        confound_csv = confoundPath +participant+ "_task-rest_desc-confounds_regressors.csv"
        confounds.append(confound_csv)
func_filenames = np.asarray(func_filenames)
stru_filenames = np.asarray(stru_filenames)
confounds= np.asarray(confounds)
#%%Loading labels
        
final_labels = np.zeros((len(labels)))
for x in range(0, len(labels)):    
    if labels[x]<14:
        final_labels[x] = 0    
    else:
        final_labels[x] = 1
numberofMDDs = len(final_labels[final_labels==1])
numberofHCs = len(final_labels[final_labels==0])

#%%Prepare the database

#CANICA-------------
n_components =10

    
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,                  
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_filenames,confounds=confounds)

# Retrieve the components
components_img = canica.components_img_

##ICA-----------
masker_ICA = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01,memory='nilearn_cache', memory_level=1)

#Timeseries-----------
correlations_ICA = []
pooled_timeseries_ICA= []
for filename, confound in zip(func_filenames, confounds):
    timeseries_each_subject = masker_ICA.fit_transform(filename,confounds=[confound] )
    pooled_timeseries_ICA.append(timeseries_each_subject)
    
#Correlations-----------
connectome_measure = ConnectivityMeasure(kind='correlation')
#connectome_measure = ConnectivityMeasure(kind='correlation', discard_diagonal=True)
#connectome_measure = ConnectivityMeasure(kind='correlation', vectorize=True )

correlations_ICA = connectome_measure.fit_transform(pooled_timeseries_ICA)


#%%#%%Create Graph
##referenec: https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/09-fcma.html#connectome
##reference: https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/09-fcma.html#connectome



import numpy as np
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
    
    for row_counter in range(epoch_corr.shape[0]):
        nodelist.append(str(row_counter))  # Set up the node names
        
        for col_counter in range(epoch_corr.shape[1]):
            
            # Determine whether to include the edge based on whether it exceeds the threshold
            if abs(epoch_corr[row_counter, col_counter]) > threshold:
                # Add a tuple specifying the voxel pairs being compared and the weight of the edge
                edgelist.append((str(row_counter), str(col_counter), {'weight': epoch_corr[row_counter, col_counter]}))
            
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

def feature_vector(G):
#    G=graph
    featureVector=[]
    #Local measures
    #--------------
    glm_l_betweennessCentrality=nx.betweenness_centrality(G) #1.betweenness centrality (defined as the fraction of all shortest paths in the network that pass through a given node.)
    glm_l_clusteringCoefficient= nx.clustering(G)#2.clustering coefficient(cc)
    glm_l_characteristicPath = dict(nx.all_pairs_shortest_path_length(G)) #3.characteristic path
    
    glm_l_csn = nx.algorithms.community.girvan_newman(G) #4.community structure Newman (CSN)
    glm_l_csn = tuple(sorted(c) for c in next(glm_l_csn))
    #glm_l_csl =community_louvain.best_partition(G) #5.community structure Louvain (CSL) # use community of 'python-louvain' vs 'networkx.algorithms.community' 
    glm_l_evcentrality = nx.eigenvector_centrality(G) #6.eigenvector centrality
    glm_l_richClubCoef = nx.rich_club_coefficient(G, normalized=False)#7.rich_club_coefficient is not implemented for graphs with self loops.
    glm_l_subGraphCentrality = nx.algorithms.centrality.subgraph_centrality(G) #8.sub graph centrality
    if nx.is_connected(G):
        glm_l_eccentricity = eccentricity(G) #9.eccentricity
    
   
    
    #The global graph measures were assortativity, clustering
    #coefficient, characteristic path, community structure Newman
    #output, community structure Louvain output, cost efficiency
    #(two measures), density, efficiency, graph radius, graph diameter,
    #transitivity, and small-worldness (45).
    #The global efficiency is the average inverse shortest path length in the network
    #The global cost efﬁciency is then deﬁned as the global efﬁciency at a given cost minus the cost,i.e.,(E	C),which will typically have a maximum value max(E	C)0,atsomecostCmax,foraneconomicalsmall-worldnetwork.Likewise,the regionalcostefﬁciencywascalculatedasthemaximumofthefunction(E(i)	k),wherekisthedegreeornumberofedgesconnectingtheith
    #Global measures
    #--------------
    glm_g_degree_assortativity_coef = nx.algorithms.assortativity.degree_assortativity_coefficient(G) # 1.assortativity
    glm_g_clusteringCoefficient=nx.average_clustering(G) #2.Global Clustering Coefficient (CC)
    glm_g_characteristicPath= nx.average_shortest_path_length(G)#3.characteristic path
    #4.community structure Newman output
    #5.community structure Louvain output
    #?glm_g_globalCostEfficiency = glm_g_global_efficiency - #6.cost efficiency(two measures)
    glm_g_density = nx.density(G)#7.density
    glm_g_global_efficiency = nx.global_efficiency(G)#8.efficiency 
    glm_g_diameter = nx.diameter(G)#9.graph diameter
    glm_g_transitivity= nx.transitivity(G)#10.transitivity
    glm_g_smallworld_sigma = nx.algorithms.smallworld.sigma(G)#11.small-worldness
    glm_g_radius = nx.radius(G)#12.graph radius

    
    featureVector=list(glm_l_betweennessCentrality.values())
    featureVector.append(list(glm_l_clusteringCoefficient.values()))
    featureVector.append(list(glm_l_characteristicPath.values()))
    featureVector.append(list(glm_l_evcentrality.values()))
    featureVector.append(list(glm_l_richClubCoef.values()))
    featureVector.append(list(glm_l_subGraphCentrality.values()))
    featureVector.append(list(glm_l_eccentricity.values()))
    
    featureVector.append(glm_g_degree_assortativity_coef)
    featureVector.append(glm_g_clusteringCoefficient)
    featureVector.append(glm_g_characteristicPath)
    featureVector.append(glm_g_density)
    featureVector.append(glm_g_global_efficiency)
    featureVector.append(glm_g_diameter)
    featureVector.append(glm_g_transitivity)
    featureVector.append(glm_g_smallworld_sigma)
    featureVector.append(glm_g_radius)
    
    return featureVector
#%%
fv_list=[]
nNodes = []
for i in range (0,correlations_ICA.shape[0]):
    
    graph = CreateGraph(correlations_ICA[i,:,:])
    nNodes.append(len(list(graph.nodes)))
    fv_list.append( feature_vector(graph))
#%%
#find an optimal subset of features
def FeatureSelection(train_X,train_Y,test_X):
    #reference:https://github.com/danielhomola/mifs
    #filter method and sorted features based on their MRMR scores    
    #The MRMR score for a feature set S
    #-----------------------------------
    import pandas as pd
    import mifs
    

    
    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()
    
    # find all relevant features
    feat_selector.fit(train_X,train_Y)
    
    # check selected features
    feat_selector.support_
    
    # check ranking of features
    feat_selector.ranking_
    
    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(train_X)
    test_filtered = feat_selector.transform(test_X)
    
    return X_filtered,test_filtered, feat_selector.ranking_
    #Create list of features
    #feature_name = train_X.columns[feat_selector.ranking_]
        
    
    #The top 50 features identified by the MRMR algorithm were used in 
    #a wrapper algorithm (a sequential features collection (SFC) algorithm) 
    # with the SVM classifier to find an optimal subset of features.
    #we sorted all featured using five feature selection algorithms, i.e.MRMR, Fisher score, Chi-square score, Gini score, and Kruskal-Wallis test. Output of each feature selection algorithm was a sortedfeature vector
    #The sorted feature vectorswere inputs of the SFC algorithm
         
    #-------------------------------------
#%%
from sklearn.model_selection import StratifiedKFold,train_test_split,GroupShuffleSplit
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix, f1_score, roc_auc_score,precision_score,recall_score
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.backend import tensorflow_backend as K

    
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,0]
#%% #KFOLD
########################

x= np.asarray(fv_list)
y =final_labels
cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
print ("StratifiedKFold:",cv)
classifier = SVC(kernel='rbf')
scores,scores1 = {},{}
for train_index, test_index in cv.split(x,y):
    print("**")
    train_X,test_X,train_Y,test_Y  = x[train_index], x[test_index], y[train_index], y[test_index]
    
    numHC = len(train_Y [train_Y ==0])
    numMDD = len(train_Y [train_Y ==1])
    print ("Train numHC,numMDD: ", numHC,",", numMDD)
    numHC = len(test_Y [test_Y ==0])
    numMDD = len(test_Y [test_Y ==1])
    print ("Test numHC,numMDD: ", numHC,",", numMDD)
    
    
    #Augmentation
    #########################
    
#    train_X_a,train_Y_a,train_confound = Augmentation(train_X,train_Y ,'duplicate',train_confound)   
#    numHC = len(train_Y_a [train_Y_a ==0])
#    numMDD = len(train_Y_a [train_Y_a ==1])
#    print ("Augmented Train numHC,numMDD: ", numHC,",", numMDD)
    
    
    #Feature Extractor
    #########################     
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    device_count={"CPU": 80}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 4,   
                intra_op_parallelism_threads = 80)) as sess:
        K.set_session(sess)
        
        #find an optimal subset of features
        train_data_features,test_data_features = FeatureSelection(train_X,train_Y,test_X)
        
           
#    train_data_features,test_data_features = Vgg16_Seq(train_X,train_Y,test_X,test_Y)
    print ("#Feature Extracted")    
    
    #Classifier
    #########################
    classifier.fit(train_data_features, train_Y)  
    preds = classifier.predict(test_data_features)
    print("#Classifier")
    scores.setdefault('test_AUC', []).append(roc_auc_score(test_Y,preds))
    scores.setdefault('test_Accuracy', []).append(accuracy_score(test_Y, preds))
    tps=tp(test_Y, preds)
    tns = tn(test_Y, preds)
    fps=fp(test_Y, preds)
    fns= fn(test_Y, preds)
    sensitivity= tps/(tps+fns)
    specificity=tns/(tns+fps)
    percision=tps/(tps+fps)
    scores.setdefault('test_sensitivity', []).append(sensitivity)
    scores.setdefault('test_specificity', []).append(specificity)
    scores.setdefault('test_percision', []).append(percision)
    scores.setdefault('test_prec', []).append(precision_score(test_Y, preds))
    scores.setdefault('test_rec', []).append(recall_score(test_Y, preds))
    scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y, classifier.predict(train_data_features)))
    scores.setdefault('test_tp', []).append(tps)
    scores.setdefault('test_tn', []).append(tns)
    scores.setdefault('test_fp', []).append(fps)
    scores.setdefault('test_fn', []).append(fns)
print(scores)