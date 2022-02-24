# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:57:02 2020

@author: mmousa4
"""
import numpy as np
import pandas as pd
import os

#%% Loading data

from nilearn.connectome import GroupSparseCovarianceCV,sym_matrix_to_vec,ConnectivityMeasure
from collections import Counter

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


os.chdir('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\NewResults')
#indx_rus = np.load('indx_rus_9_323232.npy')

final_labels = np.load('final_labels.npy')
#final_labels=final_labels[indx_rus]

print('Dataset shape %s' % Counter(final_labels))

#Features
#-----------------
#aal_GL_covariences,aal_corr_den,aal_covariance,aal_GL_graphFeatures_balanced,
#canica_GL_covariences,canica_corr_den,canica_GL_graphFeatures_balanced,
#signal_corr_4tiles_323232,tiles646464_1d
#_aal_adhdcorr_den,_aal_adhdcorr_spearmanr,_aal_adhdcorr_kendalltau,_aal_adhdGL_covariences

correlation= np.load('aal_corr_spearmanr.npy')
input_data = sym_matrix_to_vec(correlation, discard_diagonal=True)#[indx_rus]

#input_data = np.load('tiles646464_1d.npy')
#input_data= np.load('canica_GL_graphFeatures.npy')
input_data[np.isnan(input_data)==True]=0
#input_data=np.reshape(input_data,(279,-1))
#input_data = input_data[indx_rus]

#signal_corr = np.load('signal_corr_4tiles_323232.npy')
#input_data=np.reshape(signal_corr,(signal_corr.shape[0],-1))
#input_data = input_data[indx_rus]

#
#regionalFeature = np.load('aal_spectrumDensity.npy')
#regionalFeature= regionalFeature[:, :,:].reshape((regionalFeature.shape[0],-1))
#regionalFeature= regionalFeature[indx_rus]
#


#
#combFeatures = np.concatenate((regionalFeature,connectivities),axis=1)
#
#ts_aal=np.load('ts_aal.npy')
#ts_canica=np.load('ts_canica.npy')
#combFeatures_ts = np.concatenate((ts_aal,ts_canica),axis=2)

#%%Balancing the dataset
#x,y=combFeatures,final_labels
#x1=x[final_labels==1]
#x0=x[final_labels==0]
#combFeatures_balanced=np.concatenate((x1,x0[0:38]),axis=0)
#final_labels_balanced=np.concatenate(([1]*x1.shape[0],[0]*38),axis=0)
#print('Dataset balanced shape %s' % Counter(final_labels_balanced))
#%%FeatureSelection
from sklearn.feature_selection import chi2,f_classif
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import SelectKBest,RFECV
from sklearn.svm import SVR,SVC


#def FeatureSelection4():
    #relevance of the features (D) ,redundancy of the features(R)
    #The mRMR score for the set S is defined as (D - R)

def FS_Anova_RFE_RF(train_X,train_Y,test_X):
    train_X = pd.DataFrame(data=train_X)
    test_X = pd.DataFrame(data=test_X)
    #Filter method + wrapper method
    
    print("Filter method (Anova) to sort features + wrapper method")
    # Filter Methods
    #-----------------------
    #1- Chi2
    numFetaures =30
#    model = SelectKBest(score_func= chi2, k=numFetaures) 
    #1- ANOVA
    model = SelectKBest(score_func= f_classif, k=numFetaures)#chi2,f_classif,f- 

    fit = model.fit(train_X,train_Y)
    indexes = fit.get_support(True)
    features = np.asarray(train_X.columns[indexes])    
    train_features = model.transform(train_X)
    test_filtered = model.transform(test_X)

     #Wrapper Methods
    #-------------------------
     #7- RFE + RandomForestClassifier
    model=RandomForestClassifier(n_estimators=10,n_jobs=-1)
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(train_features,train_Y)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), features))
    sequence=[ seq[1] for seq in ziptw ]
    features3=sequence[0:numFetaures]
    indexes3=[features3.index(c) for c in features3]
    x_new3 = train_features[:,indexes3]
    x_new4 = test_filtered[:,indexes3]
      
    return x_new3,x_new4,indexes3

    
    
def FS_chi2_SFS(train_X,train_Y,test_X):
#    train_X,train_Y,test_X = abs(hc),train_Y[train_Y==0],test_X
    train_X = pd.DataFrame(data=train_X)
    test_X = pd.DataFrame(data=test_X)
#    Feature selection
#algorithms roughly divide into two categories: filter and
#wrapper methods. The filter methods select a subset of features #according to the general characteristics of data, independently
#of chosen classifier. However, the wrapper methods require a predetermined classifier and evaluate features according
#to their performances in discrimination of classes.
#The Fisher algorithm and the sequential feature selection algorithm
#are the most popular filter and wrapper methods, respectively.
#These algorithms are explained briefly here.
    #The wrapper methods were applied to the first half of the sorted features by the filter methods.
    
    
    # (Filter method) Calcualte the Fisher Score (chi2) between each feature and target 
    fisher_score = chi2(train_X,train_Y)
    fisher_score
    #Two arrays are returned: F scores and P value. More the P value, more difference between the distributions.
    #Smaller the p_value, more significant the feature is to predict the target value
    p_values = pd.Series(fisher_score[1])
    p_values.index = train_X.columns
    p_values.sort_values(ascending=False)
    
#   (Wrapper method)  Build RF classifier to use in feature selection
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    # Build step forward feature selection
    sfs1 = sfs(clf,
               k_features=30,
               forward=True,
               floating=False,
               verbose=0,
               scoring='accuracy',
               cv=5)
    
    # Perform SFFS
    sfs1 = sfs1.fit(train_X,train_Y)
    # Which features?
    feat_cols = list(sfs1.k_feature_idx_)
    print(feat_cols)
    return train_X.loc[:, feat_cols],test_X.loc[:, feat_cols],feat_cols


   
def FS_Fisher_SFS(train_X,train_Y,test_X):    
    train_X = pd.DataFrame(data=train_X)
    test_X = pd.DataFrame(data=test_X)
    #Fisher score for each feature is defined as , the higher the fisher score, the more discriminative feature is
    #-----------------------------
    mu = train_X.mean(axis=1) # the feature mean value of all the samples
     
    n_1 = sum(train_Y == 1) #n_1 & n_0 are the numbers of samples in the two classes
    mu_1 = train_X[train_Y == 1].mean() # the feature mean value
    var_1 = train_X[train_Y == 1].var() #variance of one class
    
    n_0 = sum(train_Y == 0)
    mu_0 = train_X[train_Y == 0].mean()
    var_0 = train_X[train_Y == 0].var()
    
    inter_class = n_1*np.power((mu_1-mu), 2) + n_0*np.power((mu_0-mu), 2)
    intra_class = (n_1)*var_1 + (n_0)*var_0
    
    fisherScore = inter_class / intra_class
    #A larger Fisher score indicates a more discriminative
    #feature. We rank all the features in the training set based
    #on Fisher score. Different feature sets can be obtained by
    #selecting different number of ordered features. The final
    #selected feature set is the one with the highest accuracy tested
    #on the validation set, which is picked out from the training
    #set
#    fisherScore = pd.Series(fisherScore)
    fisherScore = fisherScore.sort_values(ascending=False) 
    train_X_ = train_X.reindex(fisherScore.index.tolist(), axis=1)
    test_X_ =  test_X.reindex(fisherScore.index.tolist(), axis=1)

    
    #Wrapper Method 
    #--------------------
#    Forward sequential feature selection (FSFS) 
    #reference: https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/
    from mlxtend.feature_selection import SequentialFeatureSelector
    feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=30,
           forward=True,
           verbose=0,
           scoring='roc_auc',
           cv=4)    
    features = feature_selector.fit(train_X_[list(train_X_.columns[0:30])].fillna(0), train_Y)
    filtered_features= train_X_.columns[list(features.k_feature_idx_)]
    np.save('filtered_features_train.npy',train_X_[filtered_features])
    np.save('filtered_features_test.npy',test_X_[filtered_features])
    
    return train_X_[filtered_features],test_X_[filtered_features]
    

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel

def FS_Voting(x_combine, y_combine,x_test):
    Fs_methods=[]
    df_combine= pd.DataFrame(x_combine)
     
    numFetaures=30
        # Filter Methods
    #-----------------------
    #1- ANOVA
    model = SelectKBest(score_func= f_classif, k=numFetaures)#chi2,f_classif,f- 
    fit = model.fit(x_combine, y_combine)
    indexes1 = fit.get_support(True)
    features1 = np.asarray(df_combine.columns[indexes1])
    Fs_methods.append("SelectKBest_f_classif")
    x_new1 = model.transform(x_combine)    
    
    
    #2- Chi2
    model = SelectKBest(score_func= chi2, k=numFetaures) 
    fit = model.fit(x_combine, y_combine)
    indexes2 = fit.get_support(True)
    features2 = np.asarray(df_combine.columns[indexes2])
    Fs_methods.append("SelectKBest_chi2")
    x_new2 = model.transform(x_combine)
    
    #Wrapper Methods
    #-------------------------
    #3- RFE + linear SVM (C=1)
    model= SVR(kernel="linear") 
    rfe= RFECV(model, cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features3=sequence[0:numFetaures]
    indexes3=[df_combine.columns.get_loc(c) for c in features3]
    x_new3 = x_combine[:,indexes3]
    Fs_methods.append("RFECV_SVM_linear_c1")
    
    
    
    #4- RFE + linear SVM (C=100)
    model=SVC(kernel='linear', C=100)
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features4=sequence[0:numFetaures]
    indexes4=[df_combine.columns.get_loc(c) for c in features4]
    x_new4 = x_combine[:,indexes4]
    Fs_methods.append("RFECV_SVM_linear_c100")
    
    #5- RFE + GradientBoostingRegressor
    model=GradientBoostingRegressor(n_estimators=50, learning_rate=0.05)
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features5=sequence[0:numFetaures]
    indexes5=[df_combine.columns.get_loc(c) for c in features5]
    x_new5 = x_combine[:,indexes5]
    Fs_methods.append("RFECV_GradientBoostingRegressor")
    
    
    #6- RFE + RandomForestRegressor
    model=RandomForestRegressor(n_estimators=10)
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features6=sequence[0:numFetaures]
    indexes6=[df_combine.columns.get_loc(c) for c in features6]
    x_new6 = x_combine[:,indexes6]
    Fs_methods.append("RFECV_RandomforestRegressor")
    
    
    #7- RFE + RandomForestClassifier
    model=RandomForestClassifier(n_estimators=10,n_jobs=-1)
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features7=sequence[0:numFetaures]
    indexes7=[df_combine.columns.get_loc(c) for c in features7]
    x_new7 = x_combine[:,indexes7]
    Fs_methods.append("RFECV_Randomforestclassifier")
    
    #8- RFE + linear regression
    model=sklearn.linear_model.LinearRegression()
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features8=sequence[0:numFetaures]
    indexes8=[df_combine.columns.get_loc(c) for c in features8]
    x_new8 = x_combine[:,indexes8]
    Fs_methods.append("RFECV_LinearRegression")
    
    
    
    #9- RFE + logistic regression
    model=LogisticRegression()
    rfe= RFECV(model,cv=5,scoring='roc_auc')
    fit = rfe.fit(x_combine, y_combine)
    ziptw=sorted(zip(map(lambda x: round(x, 4), fit.ranking_), df_combine.columns.values))
    sequence=[ seq[1] for seq in ziptw ]
    features9=sequence[0:numFetaures]
    indexes9=[df_combine.columns.get_loc(c) for c in features9]
    x_new9 = x_combine[:,indexes9]
    Fs_methods.append("RFECV_LogisticRegression")
    
    
    #Embedded Methods
    #-------------------------
    #10- ExtraTreeClassifier
    model = ExtraTreesClassifier(n_estimators=100, max_features=numFetaures)
    fit = model.fit(x_combine, y_combine)
    importance = model.feature_importances_
    indexes10= np.argsort(importance)[::-1][0:numFetaures]
    features10= np.asarray(df_combine.columns[indexes10])
    Fs_methods.append("ExtraTreesClassifier")
    model = SelectFromModel(model, prefit=True)
    x_new10 = x_combine[:,indexes10]
    
    #11- RandomForestClassifier
    num_trees =10
    model = RandomForestClassifier(n_estimators=10, max_features=numFetaures)
    fit = model.fit(x_combine, y_combine)
    importance = model.feature_importances_
    indexes11= np.argsort(importance)[::-1][0:numFetaures]
    features11= np.asarray(df_combine.columns[indexes11])
    Fs_methods.append("RandomForestClassifier")
    model = SelectFromModel(model, prefit=True)
    x_new11 = x_combine[:,indexes11]    
    
    #%Our Method: Choose most frequent Features
    #####################################
    indexlist = [features1,features2,features3,features4,features5,features6,features7,features8,features9,features10,features11]
    flat_list = [item for sublist in indexlist for item in sublist]
    from collections import Counter
    c=Counter(flat_list)
    dic=c.most_common(10)
    highfrequecyFeatures=[dic[i][0] for i in range(len(dic))]
    frequecy=[dic[i][1] for i in range(len(dic))]
    hf_indexes=[df_combine.columns.get_loc(c) for c in highfrequecyFeatures]
    x_new_chosen = x_combine[:,hf_indexes]
    x_new_test = x_test[:,hf_indexes]
    return x_new_chosen,x_new_test
#%%

    
    
def FS_kendalltau(trainX,trainY,testX):
    #Zeng, Ling-Li, et al. "Identifying major depression using whole-brain functional connectivity: a multivariate pattern analysis." Brain 135.5 (2012): 1498-1507.
    correlation,y = trainX,trainY.copy()
    # Let xij denote the functional connectivity feature i of the jth sample and yj denote the class label of this sample ( + 1 for patients and 1 for controls)
    #provides a distribution-free test of independence between two variables to measure the relevance of each feature
    
    label=y
    label[y==0]=-1
    patients = len(y[y==1])
    controls = len(y[y==-1])
    correlation_coefficient =np.zeros((correlation.shape[1]))
    x=correlation
    
    for i in range (correlation.shape[1]):
        nc,nd=0,0
        for j in range (correlation.shape[0]):
            for k in range (j+1,correlation.shape[0]):         
            
                 if np.sign(x[j,i] -x[k,i])== np.sign(label[j]-label[k]):
                    nc=nc+1
                 if np.sign(x[j,i] -x[k,i])== -1 *np.sign(label[j]-label[k]):
                    nd=nd+1
        correlation_coefficient[i] = (nc-nd)/(patients*controls)#e number of concordant and discordant pairs
#Thus, a positive correlation coefficient i indicates that the ith functional connectivity coefficient increases in the patient group compared to the control group.
# A negative i indicates that the ith functional connectivity coefficient decreases in the patient group
# We subsequently ranked features according to
# their discriminative powers and selected those with coefficients over
# a threshold as the final feature set for classification
    
    indexes= np.argsort(abs(correlation_coefficient))
    return trainX[:,indexes],testX[:,indexes]
    
#%% Grid search CV

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier


#DecisionTreeClassifier Parameters
#-------------------
#param_grid = {'criterion':['gini','entropy'],'max_depth':[5,10,20,30,40,50,70,90,100],
#'max_leaf_nodes': [5,10,20,30,40,50,70,90,100], 'min_samples_split': list(range(2, 50,5))}
#grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)


#Guassian Process Classifier Parameters
#---------------
#Reference:https://www.kaggle.com/tbarreau/gaussian-processes-for-digits-classification
#param_grid = {'kernel':[1.0*RBF(1.0), 1.0*Matern(1.0)]}
#param_grid = {'kernel':[DotProduct(i) for i in [0.2, 0.5, 1,2,3,5]] + [Matern(i) for i in [0.2, 0.5, 1,2,3,5]]  + [RBF(i) for i in [0.2, 0.5, 1,2,3,5]]}
#grid = RandomizedSearchCV(GaussianProcessClassifier(kernel=DotProduct(1.0)), param_grid,n_jobs=4,n_iter=10,random_state=0,verbose=3)


#SVM parameters
#---------------
## defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, scoring='f1') 



#param_grid = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-3, 1e-4],
#                 'C':  [1e-40,1e-20,1e-16,1e-14,1e-10,1e-08,1e-06,1e-04, 1]},
#                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


#grid = GridSearchCV(SVC(), param_grid, cv=5)




#grid.fit(func_filenames_d,final_labels_d)
# print best parameter after tuning 
#print(grid.best_params_) 
#  
## print how our model looks after hyper-parameter tuning 
#print(grid.best_estimator_) 

#%%MLP
#reference:https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]
labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

#%%
from sklearn.cluster import KMeans,AffinityPropagation,AgglomerativeClustering,Birch

def Clustering(x_input):
    #Clustering
    #---------------------
    model = KMeans(n_clusters=2, random_state=100)
#    model = AffinityPropagation(damping=0.841)
#    model = AgglomerativeClustering(n_clusters=15)
#    model = Birch(threshold=0.01, n_clusters=2)
    
#    model.fit(x_input)
#    y_model = model.predict(x_input)
    y_model = model.fit_predict(x_input)
    centers = model.cluster_centers_
    
      
#    plt.figure(2)
#    plt.scatter(x_input[:, 0], x_input[:, 1], c=y_model, s=50, cmap='viridis')
    return centers
#%%
import sklearn
import matplotlib.pyplot as plt
def MDS(x_train,y_train):
#    x_train,y_train = connectivities[:270],final_labels
    # Center the data
    X_true=x_train[y_train ==0] #x_train[:,0:2]
    X_true=X_true-X_true.mean()
    
   # similarities = euclidean_distances(X_true)
    
    mds = sklearn.manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#    pos = mds.fit(X_true, y=None, init=None)
    
    pos = mds.fit_transform(X_true, y=None, init=None)
    
    X_false=x_train[y_train ==1] #x_train[:,0:2]
    X_false=X_false-X_false.mean()
    
    mds_false = sklearn.manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    pos_false = mds_false.fit_transform(X_false, y=None, init=None)

    x_input=np.concatenate((pos,pos_false))
    pos_label=np.zeros(pos.shape[0])
    pos_false_label=np.ones(pos_false.shape[0])
    x_input_label=np.concatenate((pos_label,pos_false_label))
    
     #plot train
    fig = plt.figure(1)
    size = 100
    #Depressed
    plt.scatter(pos_false[:, 0], pos_false[:, 1], color='C0', s=size, linewidths=0,label='Depressed')
    #Healthy Controls
    plt.scatter(pos[:, 0], pos[:, 1], color='C1', s=size, linewidths=0, label='Healthy Controls')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    Clustering(x_input)
    
    
    
    return x_input,x_input_label
def LocallyLinearEmbedding(x_train,y_train,x_test,y_test):
#    x_train,y_train = connectivities[:270],final_labels
    x_train,y_train,x_test,y_test = train_X,train_Y,test_X,test_Y
    # Center the data
    X_true=x_train[y_train ==0] #x_train[:,0:2]
    X_true=X_true-X_true.mean()
    
    X_true_test=x_test[y_test ==0] #x_train[:,0:2]
    X_true_test=X_true_test-X_true_test.mean()
    
   # similarities = euclidean_distances(X_true)
#    mds = sklearn.manifold.Isomap(n_components=2)#, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')

    mds = sklearn.manifold.LocallyLinearEmbedding(n_components=2)#, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#    mds =  decomposition.KernelPCA(n_components = 2)#, kernel = 'sigmoid')

    mds.fit(X_true)
    
    pos = mds.transform(X_true)
    pos_test = mds.transform(X_true_test)
    
    X_false=x_train[y_train ==1] #x_train[:,0:2]
    X_false=X_false-X_false.mean()
    X_false_test=x_test[y_test ==1] #x_train[:,0:2]
    X_false_test=X_false_test-X_false_test.mean()
    
    mds_false = sklearn.manifold.LocallyLinearEmbedding(n_components=2)#, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#    mds_false =  decomposition.KernelPCA(n_components = 2, kernel = 'sigmoid')
    mds_false.fit(X_false )
    
    pos_false  = mds_false.transform(X_false )
    pos_false_test = mds_false.transform(X_false_test)
    
    x_input=np.concatenate((pos,pos_false))
    pos_label=np.zeros(pos.shape[0])
    pos_false_label=np.ones(pos_false.shape[0])
    x_input_label=np.concatenate((pos_label,pos_false_label))
    
    x_input_test=np.concatenate((pos_test,pos_false_test))
    pos_label_test=np.zeros(pos_test.shape[0])
    pos_false_label_test=np.ones(pos_false_test.shape[0])
    x_input_label_test=np.concatenate((pos_label_test,pos_false_label_test))
    
    #plot train
    fig = plt.figure(1)
    size = 100
    #Depressed
    plt.scatter(pos_false[:, 0], pos_false[:, 1], color='C0', s=size, linewidths=0,label='Depressed')
    #Healthy Controls
    plt.scatter(pos[:, 0], pos[:, 1], color='C1', s=size, linewidths=0, label='Healthy Controls')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    #plot test
    fig = plt.figure(2)
    size = 100
    #Depressed
    plt.scatter(pos_false_test[:, 0], pos_false_test[:, 1], color='C0', s=size, linewidths=0,label='Depressed')
    #Healthy Controls
    plt.scatter(pos_test[:, 0], pos_test[:, 1], color='C1', s=size, linewidths=0, label='Healthy Controls')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
    
    return x_input,x_input_label,x_input_test,x_input_label_test



def LocallyLinearEmbedding1(x_train,y_train,x_test,y_test):
#    x_train,y_train = connectivities[:270],final_labels
#    x_train,y_train,x_test,y_test = train_X,train_Y,test_X,test_Y
    # Center the data
    x_train_0=x_train[y_train ==0] #x_train[:,0:2]
    x_train_0=x_train_0-x_train_0.mean()
    x_train_1=x_train[y_train ==1] #x_train[:,0:2]
    x_train_1=x_train_1-x_train_1.mean()
#    x_test_s=x_test-x_test.mean()
       
    
   # similarities = euclidean_distances(X_true)
       
    lle_0 = sklearn.manifold.LocallyLinearEmbedding(n_components=20)#, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#    mds =  decomposition.KernelPCA(n_components = 2, kernel = 'sigmoid')

    lle_0.fit(x_train_0)    
    rdim_0 = lle_0.transform(x_train_0)
#    test_0 = lle_0.transform(x_test_s)
    
    
    
    lle_1 = sklearn.manifold.LocallyLinearEmbedding(n_components=20)#, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#    mds_false =  decomposition.KernelPCA(n_components = 2, kernel = 'sigmoid')
    lle_1.fit(x_train_1 )    
    rdim_1  = lle_1.transform(x_train_1 )
#    test_1 = lle_1.transform(x_test_s)
    
#    pos_false_test = mds_false.transform(X_false_test)
    
    x_input=np.concatenate((rdim_0,rdim_1))
    hc_label=np.zeros(rdim_0.shape[0])
    mdd_label=np.ones(rdim_1.shape[0])
    x_input_label=np.concatenate((hc_label,mdd_label))
    
    x_test_01=x_test-x_test.mean()
    preds_0 = lle_0.transform(x_test_01)
    preds_1 = lle_1.transform(x_test_01)
    
    center_0 = Clustering(rdim_0)
    center_1 = Clustering(rdim_1)

    preds, label=[],[]
    for i in range (x_test_01.shape[0]):
        distp0_c0 = np.linalg.norm(preds_0[i]-center_0[0]) if np.linalg.norm(preds_0[i]-center_0[0])<np.linalg.norm(preds_0[i]-center_0[1]) else np.linalg.norm(preds_0[i]-center_0[1])
#        distp0_c1 = np.linalg.norm(preds_0[i]-center_1[0]) if np.linalg.norm(preds_0[i]-center_1[0])<np.linalg.norm(preds_0[i]-center_1[1]) else np.linalg.norm(preds_0[i]-center_1[1])
#        distp1_c0 = np.linalg.norm(preds_1[i]-center_0[0]) if np.linalg.norm(preds_1[i]-center_0[0])<np.linalg.norm(preds_1[i]-center_0[1]) else np.linalg.norm(preds_1[i]-center_0[1])
        distp1_c1 = np.linalg.norm(preds_1[i]-center_1[0]) if np.linalg.norm(preds_1[i]-center_1[0])<np.linalg.norm(preds_1[i]-center_1[1]) else np.linalg.norm(preds_1[i]-center_1[1])

        if distp0_c0<distp1_c1 :
            preds.append(preds_0[i])
            label.append(0)
        else:
            preds.append(preds_1[i])
            label.append(1)
    preds = np.array(preds)
    x_input_test = preds
    x_input_label_test = label
    #plot train
    #---------------
    fig = plt.figure(1)
    size = 100
    #Depressed
    plt.scatter(rdim_1[:, 0], rdim_1[:, 1], color='C0', s=size, linewidths=0,label='Depressed')
    #Healthy Controls
    plt.scatter(rdim_0[:, 0], rdim_0[:, 1], color='C1', s=size, linewidths=0, label='Healthy Controls')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    
   
   
    return x_input,x_input_label,x_input_test,x_input_label_test

def LocallyLinearEmbedding2(x_train,y_train,x_test,y_test):
#    x_train,y_train = connectivities[:270],final_labels
#    x_train,y_train,x_test,y_test = train_X,train_Y,test_X,test_Y
    # Center the data
    x_train=x_train-x_train.mean()
    
    
   # similarities = euclidean_distances(X_true)
    
    lle = sklearn.manifold.LocallyLinearEmbedding(n_components=10)#, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#    mds =  decomposition.KernelPCA(n_components = 2, kernel = 'sigmoid')

    lle.fit(x_train)    
    rdim = lle.transform(x_train)
    
    x_test=x_test-x_test.mean()
    rdim_test = lle.transform(x_test)
     #plot train
    #---------------
    fig = plt.figure(1)
    plt.scatter(rdim[:, 0], rdim[:, 1], c=y_train, s=100, linewidths=0)
    return rdim , rdim_test

#%% CNN-LSTM
from keras import optimizers,layers,models
from keras.callbacks import EarlyStopping

def CNN_LSTM_1D(trainX,trainY,testX) :
 #refrence: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ 
#    train_X_1d,train_Y_1d,test_X_1d = temporal_features_train,train_Y,temporal_features_test
    n_steps, n_features =trainX.shape[1],trainX.shape[2]
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv1D(filters=16, kernel_size=1, activation='relu'), input_shape=(n_steps, n_features,1)))
    model.add(layers.TimeDistributed(layers.Conv1D(filters=16, kernel_size=1, activation='relu'), input_shape=(n_steps, n_features,1)))
 
    model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
    model.add(layers.TimeDistributed(layers.Conv1D(filters=32, kernel_size=1, activation='relu'), input_shape=(n_steps, n_features,1)))
    model.add(layers.TimeDistributed(layers.Conv1D(filters=32, kernel_size=1, activation='relu'), input_shape=(n_steps, n_features,1)))
    model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))

    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(5, activation='relu'))#,return_sequences=True
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    es = EarlyStopping(monitor='loss', mode='min', verbose=0)  
#    cnn_lstm_model.fit_generator(generator=training_generator,
#        steps_per_epoch=train_steps_per_epoch, verbose=1, callbacks=callbacks,
#        validation_data=validation_generator, validation_steps=validate_steps_per_epoch,
#        epochs=epochs)
#   
    model.summary()
#    utils.print_summary(model, line_length=None, positions=[.30, .65, .77, 1.], print_fn=None)
    trainX = trainX.reshape((-1,trainX.shape[1],trainX.shape[2],1))
    testX = testX.reshape((-1,testX.shape[1],testX.shape[2],1))
    model.fit(trainX, trainY, batch_size=8,epochs=5,verbose=1,callbacks=[es])
    
    
#    model2 = models.Model(input=cnn_lstm_model.input,output=cnn_lstm_model.layers[-2].output)
#    print (model2.summary())
    train_preds = model.predict(trainX)
    test_preds= model.predict(testX)
    train_preds= np.squeeze (np.round(train_preds).astype('int'))
    test_preds= np.squeeze (np.round(test_preds).astype('int') )

    print ("#TrainFeatures shape", str(train_preds.shape)) 
    print ("#TestFeatures shape", str(test_preds.shape))   
    return train_preds, test_preds


#%%Removing outlier
from sklearn.ensemble import IsolationForest

def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = IsolationForest()
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]



#%%
#Reference:https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model, tree, discriminant_analysis, naive_bayes, ensemble, gaussian_process
from sklearn.linear_model import LogisticRegression,SGDClassifier
#from sklearn.linear_model import LogisticRegression

vote_est = [('etc',ensemble.ExtraTreesClassifier()),
            ('gb',ensemble.GradientBoostingClassifier()),
            ('abc',ensemble.AdaBoostClassifier()),
    ('rfc', ensemble.RandomForestClassifier(criterion='gini', max_depth=8, max_features='auto', n_estimators=200)),
    #('svc', svm.SVC(probability=True)),
    #('xgb', XGBClassifier()),
           # ('lbgm',LGBMClassifier())
           ]

names = ["LogisticRegression",
        "Nearest Neighbors",
#        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         
         "Bernoli_NB",
         #"Guassian_NB",
         "DiscriminantAnalysis_LDA",#"DiscriminantAnalysis_QDA",
         "Bagging","ET","GradientBoost","SGDClassifier",
         "VotingClassifier"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
    SVC(),
    GaussianProcessClassifier(),#Matern(length_scale=0.2, nu=1.5)),#(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(),#max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(),#alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    
    naive_bayes.BernoulliNB(),
    #naive_bayes.GaussianNB(),
    discriminant_analysis.LinearDiscriminantAnalysis(),
#    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    
    SGDClassifier(loss='log', tol=None, random_state=42),
    ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
    ]

#%% Classification
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix, f1_score, roc_auc_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR,SVC
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,0]

#%% T-Test
#import scipy as sp
# #2nd way
#input_data_vec_0=input_data[final_labels==0]
#input_data_vec_1=input_data[final_labels==1]
#
##t_statistic, p_value = sp.stats.ttest_ind(input_data_vec_0, input_data_vec_1, axis=0)
#statistic_value=np.zeros((input_data.shape[1]))
#p_value=np.zeros((input_data.shape[1]))
#for i in range(0,input_data.shape[1]):
##   statistic_value[i], p_value[i] = sp.stats.ttest_ind(input_data_vec_0[:,i], input_data_vec_1[:,i])
#   #statistic_value[i], p_value[i] = sp.stats.ks_2samp(input_data_vec_0[:,i], input_data_vec_1[:,i])
#   statistic_value[i], p_value[i] = sp.stats.ranksums(input_data_vec_0[:,i], input_data_vec_1[:,i])
#   #statistic_value[i], p_value[i] = sp.stats.mannwhitneyu(input_data_vec_0[:,i], input_data_vec_1[:,i])
#
#Indx_p_value_sorted = np.argsort(p_value)
#input_data_vec_sorted = input_data[:,Indx_p_value_sorted]
##input_data = input_data_vec_sorted[:,-10:]
#input_data = input_data_vec_sorted[:,0:40]


#%%K Fold CV
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut,StratifiedKFold,train_test_split,LeaveOneOut
from sklearn.neural_network import MLPClassifier
from imblearn import under_sampling ,over_sampling, FunctionSampler
from scipy.stats import ttest_ind,ranksums
from sklearn import decomposition
#-------------
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)#GroupShuffleSplit(n_splits=3)#GroupKFold()#LeaveOneGroupOut()
print ("CrossValidation:", cv)


for name, classifier in zip(names, classifiers):

    #CV LOOP
    #--------------
    scores,scores1 = {},{}
    preds_final = []
    
    #x,y=signal_spatial_1d_indv_norm, final_labels
#    x,y=func_filenames_d,final_labels_d
    
#    x,y=x_input,x_input_label
#    x,y=train_feature,final_labels_train_3d[:,0]
#    x,y=ts_canica, final_labels[:270]
#    x,y=connectivities,final_labels
#    x,y= graphFeatures,final_labels
#    x,y=input_data[indx_rus_9_pc],final_labels[indx_rus_9_pc]
    x,y=input_data,final_labels
#    x,y=combFeatures,final_labels
#    score=sklearn.model_selection.cross_val_score(classifier, x,y, cv=10)
#    print(classifier)
#    print(score)
  #%
    for train_index, test_index in cv.split(x,y):#,groups=final_labels):
#        print("split train test")
        train_X,test_X,train_Y,test_Y  = x[train_index], x[test_index], y[train_index], y[test_index]
#        train_X,test_X,train_Y,test_Y  = x.iloc[train_index], x.iloc[test_index], y[train_index], y[test_index]
        print('Dataset shape %s' % Counter(test_Y))
        
        #-------------------
#        mdd = train_X[train_Y==1]
#        hc = train_X[train_Y==0]
#        
#        atlas_mdd = np.mean(mdd, axis=0)
#        atlas_hc = np.mean(hc, axis=0)
#        
#        diff = atlas_mdd-atlas_hc
#        
#        largest_indices = np.argsort(-1*diff)[:10]
#        train_X = train_X[:,largest_indices]
#        test_X= test_X[:,largest_indices]
        
#        reject_sampler = FunctionSampler(func=outlier_rejection)
#        train_X, train_Y = reject_sampler.fit_resample(train_X, train_Y)

        
        
        #Feature Selection
        #-------------------------------
        
#        train_X,train_Y,test_X,test_Y = MDS(train_X,train_Y)
#        train_X,train_Y,test_X,preds = LocallyLinearEmbedding1(train_X,train_Y,test_X,test_Y)
#        train_X,test_X = LocallyLinearEmbedding2(train_X,train_Y,test_X,test_Y)
#        test_X,test_Y = MDS(test_X,test_Y)
        
#        model = decomposition.FastICA(n_components=2)
###        model = decomposition.KernelPCA(n_components=20)        
###        model = decomposition.PCA(n_components=30)
##        
#        model.fit(train_X)
#        train_X= model.transform(train_X)
#        test_X= model.transform(test_X)        
#        
         
#        tr1_X,te1_X,ind_1=FS_chi2_SFS(abs(mdd),train_Y[train_Y==1],test_X)
#        tr0_X,te0_X,ind_0=FS_chi2_SFS(abs(hc),train_Y[train_Y==0],test_X)
#        intersection = ind_0.union(ind_1)
#        
#        train_X,test_X=FS_Anova_RFE_RF(train_X,train_Y,test_X)
#        train_X,test_X=FS_chi2_SFS(train_X,train_Y,test_X)
#        train_X,test_X=FS_Fisher_SFS(train_X,train_Y,test_X)
#        train_X,test_X=FS_Voting(train_X,train_Y,test_X)
#        train_X,test_X=FS_kendalltau(train_X,train_Y,test_X)
#        
#        train_X,train_Y , test_X,test_Y  =LocallyLinearEmbedding2(train_X,train_Y,test_X,test_Y)

        # transform the dataset
       
        
        
        #Feature Extractor
        #########################
    #    y_test_1d,preds = LSTM_Stacked (train_X,test_X,train_Y,test_Y)
    #    y_test_1d,preds = MSRNN(train_X,test_X,train_Y,test_Y)
    #    preds_final.append([y_test_1d[0],preds[0]])
    
        #twp sample t-test
        #########################
         #Two-sample t-test
       
#        train_connectomes_1= train_X[train_Y==1]
#        train_connectomes_0= train_X[train_Y==0]
#        pvalue_lst=[]
#        for i in range(train_X.shape[1]):
#            #        tstat,pvalue = ttest_ind(train_connectomes_1,train_connectomes_0,axis=0)
#            tstat,pvalue = ranksums(train_connectomes_0[:,i],train_connectomes_1[:,i])
#            pvalue_lst.append(pvalue)
#
#        order_indx= np.argsort(np.array(pvalue_lst))
#        
#        nfeat=10
#        train_connectomes_sorted= train_X[:,order_indx]
#        train_connectomes_ = train_connectomes_sorted[:,0:nfeat]
#       
#        bests_indx = order_indx[:nfeat]
#        train_X = train_X[:,bests_indx]
#        test_X = test_X[:,bests_indx]
        
        
        
#        train_X,test_X,indx=FS_Anova_RFE_RF(train_X,train_Y,test_X)
        
         #Augmentation
        ######################
#        oversample = over_sampling.ADASYN()
        oversample = over_sampling.SMOTE()
        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
#        print("balancing")
#        rus = over_sampling.RandomOverSampler(random_state=42)
#        train_X, train_Y = rus.fit_resample(train_X, train_Y)
#
#        rus = under_sampling.RandomUnderSampler(random_state=42)
#        train_X, train_Y = rus.fit_resample(train_X, train_Y)
       
        #Classification
        #########################
        classifier.fit(train_X, train_Y)  
#        print("fit")
        preds = classifier.predict(test_X)
#        print('predict')
#        train_result, preds= CNN_LSTM_1D(train_X,train_Y,test_X)
      
#        train_result, preds = CNN(train_X,train_Y,test_X,test_Y)
#        train_result, preds = MultiHeaded_CNN(train_X,train_Y,test_X,test_Y)
        
#        train_result, preds = time_sequence_classifier(train_X,train_Y,test_X,test_Y)
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
        #    scores.setdefault('test_prec', []).append(precision_score(test_Y, preds))
        scores.setdefault('test_rec', []).append(recall_score(test_Y, preds))
        #    scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y_a, classifier.predict(train_data_flatten)))
        #    scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y, classifier.predict(train_data_flatten)))
        scores.setdefault('test_tp', []).append(tps)
        scores.setdefault('test_tn', []).append(tns)
        scores.setdefault('test_fp', []).append(fps)
        scores.setdefault('test_fn', []).append(fns)
    print (scores)
    import statistics 
#    if (statistics.mean(scores['test_sensitivity'])>0.5 and statistics.mean(scores['test_specificity'])>0.5):
    print(name,classifier)
     #    print (scores)
    print ("*Mean Performance: sensitivity,specificity,AUC,Accuracy,precision,recall")
    print([statistics.mean(scores['test_sensitivity']),statistics.mean(scores['test_specificity']),statistics.mean(scores['test_AUC']),statistics.mean(scores['test_Accuracy']),statistics.mean(scores['test_percision']),statistics.mean(scores['test_rec'])])
    print ("###########################################################")
    
#%%LOOCV
#cv = LeaveOneOut()
#from museotoolbox.cross_validation import LeaveOneSubGroupOut
#cv = LeaveOneSubGroupOut(verbose=False,random_state=12)
#print ("CrossValidation:", cv)
#
##LOSGO.get_n_splits(X,y,s)
##for tr,vl in LOSGO.split(X,y,s):
##    print(tr.shape,vl.shape)
#
##Classifier
##--------------
#classifier = SVC(kernel='rbf',C=10, degree=3, gamma=0.001)
##classifier = RandomForestClassifier()
##classifier =  MLPClassifier(solver='lbfgs', alpha=i,
##                                     random_state=1, max_iter=2000,
##                                     early_stopping=True,
##                                     hidden_layer_sizes=[100, 100])
##classifier = MLPClassifier(random_state=0,
##                            max_iter=400, **params[0])
##classifier= MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
##                    solver='sgd', verbose=10, random_state=1,
##                    learning_rate_init=.1)
##
#scores,scores1 = {},{}
#preds_final = []
##x,y=signal_spatial_1d_indv_norm, final_labels
##    x,y=func_filenames_d,final_labels_d
#x,y=connectivities[:270],final_labels[0:270]
##    x,y=np.reshape(corr_den[0:270],(270,-1)),final_labels[0:270]
#
##x,y=signal_spatial_1d_indv_norm, np.array(labels)
##    x,y=df,final_labels
##    x,y = df,final_labels
##    x,y=x_input,x_input_label
##    x,y=train_feature,final_labels_train_3d[:,0]
##    x,y=ts_canica, final_labels[:270]
#
#for train_index, test_index in cv.split(x,y,groups=final_labels):
#   
#    train_X,test_X,train_Y,test_Y  = x[train_index], x[test_index], y[train_index], y[test_index]
##        train_X,test_X,train_Y,test_Y  = x.iloc[train_index], x.iloc[test_index], y[train_index], y[test_index]
#    
#    
#    
#    #Feature Selection
#    #-------------------------------
#    
#
#    # transform the dataset
#    #Augmentation
#    ######################
##        oversample = ADASYN()
#    oversample = SMOTE()
#    train_X, train_Y = oversample.fit_resample(train_X, train_Y)
##        
#    
#    #Feature Extractor
#    #########################
##    y_test_1d,preds = LSTM_Stacked (train_X,test_X,train_Y,test_Y)
##    y_test_1d,preds = MSRNN(train_X,test_X,train_Y,test_Y)
##    preds_final.append([y_test_1d[0],preds[0]])
#    #
#    
#    #Classification
#    #########################
#    classifier.fit(train_X, train_Y)  
#    preds = classifier.predict(test_X)
#    preds_final.append([test_Y,preds])
#
#preds_final=np.asarray(preds_final)
#test_Y = preds_final[:,0,0]
#preds=preds_final[:,1,0]
#scores.setdefault('test_AUC', []).append(roc_auc_score(test_Y,preds))
#scores.setdefault('test_Accuracy', []).append(accuracy_score(test_Y, preds))
#tps=tp(test_Y, preds)
#tns = tn(test_Y, preds)
#fps=fp(test_Y, preds)
#fns= fn(test_Y, preds)
#sensitivity= tps/(tps+fns)
#specificity=tns/(tns+fps)
#percision=tps/(tps+fps)
#scores.setdefault('test_sensitivity', []).append(sensitivity)
#scores.setdefault('test_specificity', []).append(specificity)
#scores.setdefault('test_percision', []).append(percision)
##    scores.setdefault('test_prec', []).append(precision_score(test_Y, preds))
#scores.setdefault('test_rec', []).append(recall_score(test_Y, preds))
##    scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y_a, classifier.predict(train_data_flatten)))
##    scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y, classifier.predict(train_data_flatten)))
#scores.setdefault('test_tp', []).append(tps)
#scores.setdefault('test_tn', []).append(tns)
#scores.setdefault('test_fp', []).append(fps)
#scores.setdefault('test_fn', []).append(fns)
#print(scores)
