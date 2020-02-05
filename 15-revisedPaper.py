# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:03:37 2018

@author: moosa
"""
#%% Data

import numpy as np
import pandas as pd



import os
os.chdir( 'D:\\Research\\fMRI\\Code\\Goal1-Volumetric Features\\' )
cwd = os.getcwd()
#
#x_train = np.load('./Data/x_train.npy')
#y_train = np.load('./Data/y_train.npy')
#x_test = np.load('./Data/x_test.npy')
#y_test = np.load('./Data/y_test.npy')


df_train = pd.read_csv("df_train_All.csv",index_col='Unnamed: 0')#D:\Research\fMRI\Code
trainData = df_train.iloc[:, 1:-2].values
trainLabel = df_train.iloc[:, -1].values

df_test = pd.read_csv("df_test_All.csv",index_col='Unnamed: 0')#D:\Research\fMRI\Code
testData = df_test.iloc[:, 1:-2].values
testLabel = df_test.iloc[:, -1].values


#%%
df_train.set_index('index',inplace=True)
df_test.set_index('index',inplace=True)

df_x_train= df_train.div(df_train["EstimatedTotalIntraCranialVol"].values, axis=0)
df_x_train = df_x_train*df_x_train.std(axis=0)/df_x_train.mean(axis=0)
df_train_label=df_train["BDI_Total"].values

df_x_test= df_test.div(df_test["EstimatedTotalIntraCranialVol"].values, axis=0)
df_x_test = df_x_test*df_x_test.std(axis=0)/df_x_test.mean(axis=0)
df_test_label=df_test["DBDI_22"].values

#%%
#if 'Left-WM-hypointensities'and'Right-WM-hypointensities'and 'non-WM-hypointensities'and 'Left-non-WM-hypointensities'and'Right-non-WM-hypointensities'and'5th-Ventricle' in df_test:
#    df_test = df_test.drop(['Left-WM-hypointensities','Right-WM-hypointensities','non-WM-hypointensities','Left-non-WM-hypointensities','Right-non-WM-hypointensities','5th-Ventricle'], axis=1)
#    df_train = df_train.drop(['Left-WM-hypointensities','Right-WM-hypointensities','non-WM-hypointensities','Left-non-WM-hypointensities','Right-non-WM-hypointensities','5th-Ventricle'], axis=1)


#%%Normalization

standardValue =  df_train["EstimatedTotalIntraCranialVol"].values
#x_train1=trainData/standardValue[:,None]
x_train=trainData/standardValue[:,None]
#x_train = (x_train-x_train.mean(axis=0))/x_train.std(axis=0)
#x_train = (x_train-x_train.min(axis=0))/(x_train.max(axis=0)-x_train.min(axis=0))
x_train = x_train*x_train.std(axis=0)/x_train.mean(axis=0)
#x_train = x_train/x_train.mean(axis=0)
#x_train = x_train/np.sqrt(np.sum(x_train**2,axis=0))

test_standardValue =  df_test["EstimatedTotalIntraCranialVol"].values
x_test=testData/test_standardValue[:,None]

#x_test = (x_test-x_test.mean(axis=0))/x_test.std(axis=0)
#x_test = (x_test-x_test.min(axis=0))/(x_test.max(axis=0)-x_test.min(axis=0))
x_test = x_test*x_test.std(axis=0)/x_test.mean(axis=0)
#x_test = x_test/x_test.mean(axis=0)
#x_test = x_test/np.sqrt(np.sum(x_test**2,axis=0))


#%%
y_train=np.zeros((len(trainLabel)))
y_test=np.zeros((len(testLabel)))

for x in range(0, len(trainLabel)):
    if trainLabel[x]<11:
        y_train[x] = 0   
        df_x_train.iloc[x]['BDI_Total']=0
        
    else:
        y_train[x] =1
        df_x_train.iloc[x]['BDI_Total']=1
        

for x in range(0, len(testLabel)):
    if testLabel[x]<11:
        y_test[x] = 0
        df_x_test.iloc[x]['DBDI_22']=0
    else:
        y_test[x] =1
        df_x_test.iloc[x]['DBDI_22']=1
#Labeling 3 classes
#0–10: indicates minimal depression
#11–23: indicates moderate depression
#24–63: indicates severe depression.

#y_train=np.zeros((len(trainLabel)))
#y_test=np.zeros((len(testLabel)))
#for x in range(0, len(trainLabel)):
#    if trainLabel[x]<11:
#        y_train[x] = 0
#    elif trainLabel[x]<24:
#        y_train[x] = 1  
#    else:
#        y_train[x] =2
#
#for x in range(0, len(testLabel)):
#    if testLabel[x]<11:
#        y_test[x] = 0
#    elif testLabel[x]<24:
#        y_test[x] = 1  
#    else:
#        y_test[x] =2

#%% Combining base Train and Test dataset 

combineData= np.vstack((trainData,testData))
x_combine = np.vstack((x_train,x_test))
#df_train['index'] = df_train['index'].astype(str)
df_combine =df_train.merge(df_test,how='outer')
df_combine = df_combine.drop(['EstimatedTotalIntraCranialVol','BDI_Total', 'DBDI_22'], axis=1)
#df_combine = df_combine.drop(['index','EstimatedTotalIntraCranialVol','BDI_Total'], axis=1) # for all features

#df_combine = pd.DataFrame(df_train)
#df_combine.append(df_test,ignore_index=True)

y_combine = np.hstack((y_train,y_test))
y_combine_real = np.hstack((trainLabel,testLabel))

#%%Test
#a=df_train['Right-Pallidum'].values
#b=df_train.columns.get_loc('Right-Pallidum')
#
##c=df_test.loc[['A00008326']]['Right-Pallidum']
#d=df_test.iloc[[0,1,2,3,4,5]][['Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','EstimatedTotalIntraCranialVol','DBDI_22']]
#d1=df_x_test.iloc[[0,1,2,3,4,5]][['Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','EstimatedTotalIntraCranialVol','DBDI_22']]

#%% Feature selection

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2,f_classif,mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#from yellowbrick.features import RFECV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer,confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score,GroupKFold,cross_validate,train_test_split,LeaveOneOut,StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier


Fs_methods=[] 
FeatureMethodAUC=[]
numFeaturesList = [10,20,30,40,50,60,70,80]
for numFetaures in numFeaturesList:

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
    model=LinearRegression()
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
    
    
    #Feature Extraction
    #--------------------------
    ##12- PCA
    #rfe = PCA(n_components=10)
    #fit = rfe.fit(x_combine, y_combine)
    #Fs_methods.append("PCA")
    #x_new12 = rfe.transform(x_combine)
    #
    #
    ##13- Autoencoder
    #from keras.layers import Input, Dense
    #from keras.models import Model
    #from keras.regularizers import l2
    #import sklearn
    #import sklearn.svm
    #from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,average_precision_score, roc_auc_score,precision_score,recall_score
    #from sklearn.model_selection import StratifiedKFold
    #
    #cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    #split,k = cv.split(x_combine,y_combine),cv.get_n_splits(x_combine,y_combine)
    #
    #f1_1,accuracy_1, auc_1,precision_1,recall_1 = [],[],[],[],[]
    #k=0
    ##%class_weight
    ##----------
    ##classifier = SVC(kernel='rbf',  probability=True, class_weight={1: 4.52})
    ##classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7, class_weight={1: 4.52} )
    #
    ##%no class_weight
    ##----------
    ##classifier = SVC(kernel='rbf',  probability=True)
    #classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7 )
    #
    ##Baseline
    ##---------
    ##classifier = DummyClassifier(strategy="most_frequent")
    ##classifier = SVC(kernel='rbf',  probability=True)
    ##classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7 )
    #
    #
    #for train, test in split:
    #    x_train_split,y_train_split, x_test_split,y_test_split= x_combine[train], y_combine[train],x_combine[test], y_combine[test]
    #    
    #    if (k==0):
    #        encoding_dim = 10 # Define Dimention of encoder 
    #        ncol=len(x_train_split[1])
    #        
    #        input_dim = Input(shape=(ncol,))
    #        # Define the encoder layer
    #        encoded = Dense(encoding_dim, activation='relu')(input_dim) 
    #        # Define the decoder layer
    #        decoded = Dense(ncol, activation='relu')(encoded)
    #        
    #        encoder = Model(input_dim, encoded)
    #        # Combine encoder and decoder into an autoencoder model 
    #        autoencoder = Model(input= input_dim, output= decoded)
    #        
    #        # configure and train the autoencoder
    #        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
    #           #'hinge'
    #            
    #        fit = autoencoder.fit(x_train_split, x_train_split,
    #                        epochs=50,
    #                        batch_size=12,
    #                        shuffle=True,
    #                        validation_data=(x_test_split, x_test_split), verbose=0)
    #        Fs_methods.append("Autoencoder")
    #    else:
    #        k = k+1
    #    x_new13 =  encoder.predict(x_train_split)
    #    test_rep = encoder.predict(x_test_split)
    #
    ##Oversampling
    ##----------
    #    rus = RandomOverSampler(random_state=42)
    #    x_resampled13, y_resampled13 = rus.fit_sample(x_new13, y_train_split)
    #  
    #    
    #    #class_weight
    #    #----------
    ##    x_resampled13, y_resampled13 =x_new13, y_train_split
    #    
    #    classifier.fit(x_resampled13, y_resampled13)
    #    preds = classifier.predict(test_rep)
    #    accuracy_1.append (accuracy_score(y_test_split, preds))
    #    auc_1.append( roc_auc_score(y_test_split,preds))
    #    f1_1.append(f1_score(y_test_split,preds) )
    #    precision_1.append(precision_score(y_test_split, preds))
    #    recall_1.append( recall_score(y_test_split, preds))
    
    #%%Our Method: Choose most frequent Features
    indexlist = [features1,features2,features3,features4,features5,features6,features7,features8,features9,features10,features11]
    flat_list = [item for sublist in indexlist for item in sublist]
    from collections import Counter
    c=Counter(flat_list)
    dic=c.most_common(10)
    highfrequecyFeatures=[dic[i][0] for i in range(len(dic))]
    frequecy=[dic[i][1] for i in range(len(dic))]
    hf_indexes=[df_combine.columns.get_loc(c) for c in highfrequecyFeatures]
    x_new_chosen = x_combine[:,hf_indexes]
    
    #%%Classifier
    ###%no class-weight
    ###----------
      
    #classifier = SVC(kernel='rbf',  probability=True)
    #classifier = SVC(kernel='sigmoid',  probability=True)
     
    classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7 )
    
    
    #from sklearn.naive_bayes import GaussianNB
    #classifier = GaussianNB() #naive Bayesian
    
    #classifier= LogisticRegression()
    
    #from sklearn.neighbors import KNeighborsClassifier  
    #classifier = KNeighborsClassifier(n_neighbors=5)  
    
    #from sklearn.tree import DecisionTreeClassifier
    ##classifier = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    #
    #from sklearn.ensemble import AdaBoostClassifier
    #classifier=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    #                         algorithm="SAMME",
    #                         n_estimators=200)
    #%% To validate the model use cross validation
    #tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,1]
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,0]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,1]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,0]
    
    scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
               'fp' : make_scorer(fp), 'fn' : make_scorer(fn)
               ,'prec': 'precision','rec':'recall','f1': 'f1','Accuracy': make_scorer(accuracy_score),'AUC': 'roc_auc'}
    
    #%% Sampling Method
    
    def OversamplingBothClasses(numtotalimage,train_data,train_ground):
    #    train_data,train_ground = x_new1, y_combine
    #    numtotalimage =2000
        train_data_oversample,train_ground_oversample,train_index_oversample=[],[],[]
        n_sample_1=len(train_ground[train_ground==1])
        n_sample_0=len(train_ground[train_ground==0])
        train_data_1 = train_data[train_ground==1]
        train_data_0 = train_data[train_ground==0]
        
        total_sample_1=int(numtotalimage/2)
        total_sample_0=int(numtotalimage/2)
            
        if total_sample_0>n_sample_0:
        
            #n_rpt=(n_sample_0-n_sample_1)
            n_rpt_1=(total_sample_1-n_sample_1)
            n_rpt_0=(total_sample_0-n_sample_0)
            
            indx_rnd_1=np.random.randint(0,n_sample_1,n_rpt_1)
            indx_rnd_0=np.random.randint(0,n_sample_0,n_rpt_0)
            
            train_data_rpt_1=[]
            train_data_rpt_1=train_data_1[indx_rnd_1,...].copy()
            
            train_data_rpt_0=[]
            train_data_rpt_0=train_data_0[indx_rnd_0,...].copy()
            
            train_data_oversample=np.concatenate((train_data,train_data_rpt_0,train_data_rpt_1),axis=0)
            train_ground_oversample=np.concatenate((train_ground,[0]*n_rpt_0,[1]*n_rpt_1),axis=0)
            train_index_oversample = np.concatenate((range(len(train_ground)),indx_rnd_0,indx_rnd_1),axis=0)
            
    #        print (len(train_ground_oversample[train_ground_oversample==1]))
    #        print (len(train_ground_oversample[train_ground_oversample==0]))
    #        
        return train_data_oversample.copy(), train_ground_oversample.copy(),train_index_oversample
    
    #%% Method1: RandomOverSampling + CV
    #rus = RandomOverSampler(random_state=42)
    #x_resampled1, y_resampled1 = rus.fit_sample(x_new1, y_combine)
    #x_resampled2, y_resampled2 = rus.fit_sample(x_new2, y_combine)
    #x_resampled3, y_resampled3 = rus.fit_sample(x_new3, y_combine)
    #x_resampled4, y_resampled4 = rus.fit_sample(x_new4, y_combine)
    #x_resampled5, y_resampled5 = rus.fit_sample(x_new5, y_combine)
    #x_resampled6, y_resampled6 = rus.fit_sample(x_new6, y_combine)
    #x_resampled7, y_resampled7 = rus.fit_sample(x_new7, y_combine)
    #x_resampled8, y_resampled8 = rus.fit_sample(x_new8, y_combine)
    #x_resampled9, y_resampled9 = rus.fit_sample(x_new9, y_combine)
    #x_resampled10, y_resampled10 = rus.fit_sample(x_new10, y_combine)
    #x_resampled11, y_resampled11 = rus.fit_sample(x_new11, y_combine)
    #x_resampled_chosen, y_resampled_chosen = rus.fit_sample(x_new_chosen, y_combine)
    
    #%% Method1: SMOTE,ADASYN OVERSAMPLING + CV
    #from imblearn.over_sampling import SMOTE,ADASYN
    #rus = SMOTE(random_state=100)#, ratio = 1.0
    ##rus = ADASYN()
    #x_resampled1, y_resampled1 = rus.fit_sample(x_new1, y_combine)
    #x_resampled2, y_resampled2 = rus.fit_sample(x_new2, y_combine)
    #x_resampled3, y_resampled3= rus.fit_sample(x_new3, y_combine)
    #x_resampled4, y_resampled4= rus.fit_sample(x_new4, y_combine)
    #x_resampled5, y_resampled5 = rus.fit_sample(x_new5, y_combine)
    #x_resampled6, y_resampled6 = rus.fit_sample(x_new6, y_combine)
    #x_resampled7, y_resampled7 = rus.fit_sample(x_new7, y_combine)
    #x_resampled8, y_resampled8 = rus.fit_sample(x_new8, y_combine)
    #x_resampled9, y_resampled9 = rus.fit_sample(x_new9, y_combine)
    #x_resampled10, y_resampled10 = rus.fit_sample(x_new10, y_combine)
    #x_resampled11, y_resampled11 = rus.fit_sample(x_new11, y_combine)
    #x_resampled_chosen, y_resampled_chosen = rus.fit_sample(x_new_chosen, y_combine)
    #
    #%% Cross validation for SMOTE, or ADASYN Oversampling
    #from sklearn.model_selection import StratifiedKFold
    #
    #cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    #scores1 = cross_validate(classifier, x_resampled1, y_resampled1,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores2 = cross_validate(classifier, x_resampled2, y_resampled2,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores3 = cross_validate(classifier, x_resampled3, y_resampled3,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores4 = cross_validate(classifier, x_resampled4, y_resampled4,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores5 = cross_validate(classifier, x_resampled5, y_resampled5,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores6 = cross_validate(classifier, x_resampled6, y_resampled6,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores7 = cross_validate(classifier, x_resampled7, y_resampled7,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores8 = cross_validate(classifier, x_resampled8, y_resampled8,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores9 = cross_validate(classifier, x_resampled9, y_resampled9,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores10 = cross_validate(classifier, x_resampled10, y_resampled10,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores11 = cross_validate(classifier, x_resampled11, y_resampled11,cv = cv, scoring=scoring,  return_train_score=True)      
    #scores_chosen= cross_validate(classifier, x_resampled_chosen, y_resampled_chosen,cv = cv, scoring=scoring,  return_train_score=True)       
    
    #%% Method2: RandomOverSampling return Groups
    
    #rus = RandomOverSampler(random_state=42,return_indices=True)
    #x_resampled1, y_resampled1, index1 = rus.fit_sample(x_new1, y_combine)
    #x_resampled2, y_resampled2, index2 = rus.fit_sample(x_new2, y_combine)
    #x_resampled3, y_resampled3, index3 = rus.fit_sample(x_new3, y_combine)
    #x_resampled4, y_resampled4, index4 = rus.fit_sample(x_new4, y_combine)
    #x_resampled5, y_resampled5, index5 = rus.fit_sample(x_new5, y_combine)
    #x_resampled6, y_resampled6, index6 = rus.fit_sample(x_new6, y_combine)
    #x_resampled7, y_resampled7, index7 = rus.fit_sample(x_new7, y_combine)
    #x_resampled8, y_resampled8, index8 = rus.fit_sample(x_new8, y_combine)
    #x_resampled9, y_resampled9, index9 = rus.fit_sample(x_new9, y_combine)
    #x_resampled10, y_resampled10, index10 = rus.fit_sample(x_new10, y_combine)
    #x_resampled11, y_resampled11, index11 = rus.fit_sample(x_new11, y_combine)
    #x_resampled_chosen, y_resampled_chosen, index_chosen = rus.fit_sample(x_new_chosen, y_combine)
    
    #%%  Method2-1: RandomOverSampling return Groups without feature selection
    numtotalimage = 800
    x_resampled1, y_resampled1, index1 = OversamplingBothClasses(numtotalimage,x_new1, y_combine)
    x_resampled2, y_resampled2, index2 = OversamplingBothClasses(numtotalimage,x_new2, y_combine)
    x_resampled3, y_resampled3, index3 = OversamplingBothClasses(numtotalimage,x_new3, y_combine)
    x_resampled4, y_resampled4, index4 = OversamplingBothClasses(numtotalimage,x_new4, y_combine)
    x_resampled5, y_resampled5, index5 = OversamplingBothClasses(numtotalimage,x_new5, y_combine)
    x_resampled6, y_resampled6, index6 = OversamplingBothClasses(numtotalimage,x_new6, y_combine)
    x_resampled7, y_resampled7, index7 = OversamplingBothClasses(numtotalimage,x_new7, y_combine)
    x_resampled8, y_resampled8, index8 = OversamplingBothClasses(numtotalimage,x_new8, y_combine)
    x_resampled9, y_resampled9, index9 = OversamplingBothClasses(numtotalimage,x_new9, y_combine)
    x_resampled10, y_resampled10, index10 = OversamplingBothClasses(numtotalimage,x_new10, y_combine)
    x_resampled11, y_resampled11, index11 = OversamplingBothClasses(numtotalimage,x_new11, y_combine)
    x_resampled_chosen, y_resampled_chosen, index_chosen = OversamplingBothClasses(numtotalimage,x_new_chosen, y_combine)
    
    #%% Group cross validation
    #group_k_fold = GroupKFold(n_splits=10)
    #cv1=list(group_k_fold.split(x_resampled1, y_resampled1,index1))
    #cv2=list(group_k_fold.split(x_resampled2, y_resampled2,index2))
    #cv3=list(group_k_fold.split(x_resampled3, y_resampled3,index3))
    #cv4=list(group_k_fold.split(x_resampled4, y_resampled4,index4))
    #cv5=list(group_k_fold.split(x_resampled5, y_resampled5,index5))
    #cv6=list(group_k_fold.split(x_resampled6, y_resampled6,index6))
    #cv7=list(group_k_fold.split(x_resampled7, y_resampled7,index7))
    #cv8=list(group_k_fold.split(x_resampled8, y_resampled8,index8))
    #cv9=list(group_k_fold.split(x_resampled9, y_resampled9,index9))
    #cv10=list(group_k_fold.split(x_resampled10, y_resampled10,index10))
    #cv11=list(group_k_fold.split(x_resampled11, y_resampled11,index11))
    #cv_chosen=list(group_k_fold.split(x_resampled_chosen, y_resampled_chosen,index_chosen))
    #
    #
    #scores1 = cross_validate(classifier, x_resampled1, y_resampled1,cv = cv1, scoring=scoring,  return_train_score=True)      
    #scores2 = cross_validate(classifier, x_resampled2, y_resampled2,cv = cv2, scoring=scoring,  return_train_score=True)      
    #scores3 = cross_validate(classifier, x_resampled3, y_resampled3,cv = cv3, scoring=scoring,  return_train_score=True)      
    #scores4 = cross_validate(classifier, x_resampled4, y_resampled4,cv = cv4, scoring=scoring,  return_train_score=True)      
    #scores5 = cross_validate(classifier, x_resampled5, y_resampled5,cv = cv5, scoring=scoring,  return_train_score=True)      
    #scores6 = cross_validate(classifier, x_resampled6, y_resampled6,cv = cv6, scoring=scoring,  return_train_score=True)      
    #scores7 = cross_validate(classifier, x_resampled7, y_resampled7,cv = cv7, scoring=scoring,  return_train_score=True)      
    #scores8 = cross_validate(classifier, x_resampled8, y_resampled8,cv = cv8, scoring=scoring,  return_train_score=True)      
    #scores9 = cross_validate(classifier, x_resampled9, y_resampled9,cv = cv9, scoring=scoring,  return_train_score=True)      
    #scores10 = cross_validate(classifier, x_resampled10, y_resampled10,cv = cv10, scoring=scoring,  return_train_score=True)      
    #scores11 = cross_validate(classifier, x_resampled11, y_resampled11,cv = cv11, scoring=scoring,  return_train_score=True)      
    #scores_chosen= cross_validate(classifier, x_resampled_chosen, y_resampled_chosen,cv = cv_chosen, scoring=scoring,  return_train_score=True)       
    
    #%%Method3- FS_RandomOversampling_CV_split train_test(no overlap)
    
    def CV_LeaveTestOut(X,Y):
        from sklearn.metrics import roc_auc_score,recall_score,precision_score
        from sklearn.model_selection import StratifiedKFold
        numtotalimage = 800
        lpo = StratifiedKFold(10)
        scores = {}
    
        rus = RandomOverSampler(random_state=42,return_indices=True) 
        x_resampled, y_resampled, indexes = rus.fit_sample(X,Y)  #Base Train , test: (620,) (68,) ; Train , test size: (255,) (257,) (34,) (34,)
    
#        x_resampled, y_resampled, indexes = OversamplingBothClasses(numtotalimage,X, Y) #Base Train , test: (720,) (80,) ; Train , test size: (291,) (94,) (40,) (40,)
    
        
        for train, test in lpo.split(x_resampled, y_resampled):        
            print("Base Train , test:",train.shape,test.shape)
            trainIndexestoRemove=[]
            for i in range(train.shape[0]):
    #            print("train element:",indexes[i])
                for j in range(test.shape[0]):                
                    if indexes[i]==indexes[j]:
    #                    print("test element:",indexes[j])
                        trainIndexestoRemove.append(i)
    #        print("removeIndexesfromTrain:" ,trainIndexestoRemove)
            
            train = np.delete(train,trainIndexestoRemove)            
    #        print("new Train , test:",train.shape,test.shape)            
    #        print("Indexes of Train , test:",np.intersect1d(indexes[train],indexes[test]))            
            train_ind = train
            test_ind = test
    #        train_ind = np.where(np.in1d(indexes, train))
    #        test_ind = np.where(np.in1d(indexes, test))
    #        print("Indx Train , test:",len(list(train_ind)[0]),len(list(test_ind)[0]))
            trainX,trainY,testX,testY= x_resampled[train_ind], y_resampled[train_ind],x_resampled[test_ind], y_resampled[test_ind]  
            
            print ("Train , test size: "+ str(trainY[trainY==0].shape),str(trainY[trainY==1].shape), str(testY[testY==0].shape), str(testY[testY==1].shape))
    
            classifier.fit(trainX, trainY)
            preds = classifier.predict(testX)
            scores.setdefault('test_AUC', []).append(roc_auc_score(testY,preds))
            scores.setdefault('test_Accuracy', []).append(accuracy_score(testY, preds))
            scores.setdefault('test_tp', []).append(tp(testY, preds))
            scores.setdefault('test_tn', []).append(tn(testY, preds))
            scores.setdefault('test_fp', []).append(fp(testY, preds))
            scores.setdefault('test_fn', []).append(fn(testY, preds))
            scores.setdefault('test_prec', []).append(precision_score(testY, preds))
            scores.setdefault('test_rec', []).append(recall_score(testY, preds))
            scores.setdefault('train_Accuracy', []).append(accuracy_score(trainY, classifier.predict(trainX)))
        
        for key, value in scores.items():
            scores[key] = np.array(value)
    #        print(key, str(scores[key]))
        return scores
    
    scores1 = CV_LeaveTestOut(x_new1,y_combine)
    scores2 = CV_LeaveTestOut(x_new2,y_combine)
    scores3 = CV_LeaveTestOut(x_new3,y_combine)
    scores4 = CV_LeaveTestOut(x_new4,y_combine)
    scores5 = CV_LeaveTestOut(x_new5,y_combine)
    scores6 = CV_LeaveTestOut(x_new6,y_combine)
    scores7 = CV_LeaveTestOut(x_new7,y_combine)
    scores8 = CV_LeaveTestOut(x_new8,y_combine)
    scores9 = CV_LeaveTestOut(x_new9,y_combine)
    scores10 = CV_LeaveTestOut(x_new10,y_combine)
    scores11 = CV_LeaveTestOut(x_new11,y_combine)
    scores_chosen= CV_LeaveTestOut(x_new_chosen,y_combine)
    
    #%%Method4.Oversampling in each fold 
    #
    #def CV_Oversample(xtrain,y_combine):
    #    
    #    from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,average_precision_score, roc_auc_score,precision_score,recall_score
    #    from sklearn.model_selection import StratifiedKFold
    #    from imblearn.over_sampling import SMOTE,ADASYN
    #    
    #    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    #    split,k = cv.split(xtrain,y_combine),cv.get_n_splits(xtrain,y_combine)
    #    
    #    scores = {}
    #    from sklearn.model_selection import GridSearchCV
    #    for train, test in split:      
    #        trainX,trainY, testX,testY= xtrain[train], y_combine[train],xtrain[test], y_combine[test]    
    #        print ("Train , test size: "+ str(trainY.shape), str(testY[testY==1].shape))
    #        #oversample train set_ No class weight
    #        #-----------------------
    ##        rus = RandomOverSampler(random_state=42,return_indices=True)
    ##        x_resampled, y_resampled, indexes = rus.fit_sample(trainX,trainY)
    #
    ##        sm = SMOTE(random_state=100)#, ratio = 1.0
    ##        sm = ADASYN()
    ##        x_resampled, y_resampled = sm.fit_sample(trainX,trainY)
    #        
    #        x_resampled, y_resampled, data_indexes = OversamplingBothClasses(800,trainX,trainY)
    ##        x_resampled_test, y_resampled_test, data_indexes_test = OversamplingBothClasses(80,testX,testY)
    #        x_resampled_test, y_resampled_test= testX,testY
    #        
    #        # fit a model
    #        classifier.fit(x_resampled, y_resampled)
    #        
    #        preds = classifier.predict(x_resampled_test)
    #        
    #        
    #        scores.setdefault('test_AUC', []).append(roc_auc_score(y_resampled_test,preds))
    #        scores.setdefault('test_Accuracy', []).append(accuracy_score(y_resampled_test, preds))
    #        scores.setdefault('test_tp', []).append(tp(y_resampled_test, preds))
    #        scores.setdefault('test_tn', []).append(tn(y_resampled_test, preds))
    #        scores.setdefault('test_fp', []).append(fp(y_resampled_test, preds))
    #        scores.setdefault('test_fn', []).append(fn(y_resampled_test, preds))
    #        scores.setdefault('test_prec', []).append(precision_score(y_resampled_test, preds))
    #        scores.setdefault('test_rec', []).append(recall_score(y_resampled_test, preds))
    #        scores.setdefault('train_Accuracy', []).append(accuracy_score(y_resampled, classifier.predict(x_resampled)))
    # 
    #        
    #    for key, value in scores.items():
    #        scores[key] = np.array(value)
    #
    #    return scores
    #
    #
    #scores1 = CV_Oversample(x_new1,y_combine)
    #scores2 = CV_Oversample(x_new2,y_combine)
    #scores3 = CV_Oversample(x_new3,y_combine)
    #scores4 = CV_Oversample(x_new4,y_combine)
    #scores5 = CV_Oversample(x_new5,y_combine)
    #scores6 = CV_Oversample(x_new6,y_combine)
    #scores7 = CV_Oversample(x_new7,y_combine)
    #scores8 = CV_Oversample(x_new8,y_combine)
    #scores9 = CV_Oversample(x_new9,y_combine)
    #scores10 = CV_Oversample(x_new10,y_combine)
    #scores11 = CV_Oversample(x_new11,y_combine)
    #scores_chosen= CV_Oversample(x_new_chosen,y_combine)
    
    #%% Method1-3: RandomOverSampling return Groups without feature selection
    
    #rus = RandomOverSampler(random_state=42,return_indices=True)
    #x_resampled1, y_resampled1, index1 = rus.fit_sample(x_combine, y_combine)
    #x_resampled2, y_resampled2, index2 = rus.fit_sample(x_combine, y_combine)
    #x_resampled3, y_resampled3, index3 = rus.fit_sample(x_combine, y_combine)
    #x_resampled4, y_resampled4, index4 = rus.fit_sample(x_combine, y_combine)
    #x_resampled5, y_resampled5, index5 = rus.fit_sample(x_combine, y_combine)
    #x_resampled6, y_resampled6, index6 = rus.fit_sample(x_combine, y_combine)
    #x_resampled7, y_resampled7, index7 = rus.fit_sample(x_combine, y_combine)
    #x_resampled8, y_resampled8, index8 = rus.fit_sample(x_combine, y_combine)
    #x_resampled9, y_resampled9, index9 = rus.fit_sample(x_combine, y_combine)
    #x_resampled10, y_resampled10, index10 = rus.fit_sample(x_combine, y_combine)
    #x_resampled11, y_resampled11, index11 = rus.fit_sample(x_combine, y_combine)
    #x_resampled_chosen, y_resampled_chosen, index_chosen = rus.fit_sample(x_combine, y_combine)
    
    
    #%%Method 2:Class_weight
    #x_resampled1, y_resampled1 = x_new1, y_combine
    #x_resampled2, y_resampled2 = x_new2, y_combine
    #x_resampled3, y_resampled3 = x_new3, y_combine
    #x_resampled4, y_resampled4 = x_new4, y_combine
    #x_resampled5, y_resampled5 = x_new5, y_combine
    #x_resampled6, y_resampled6 = x_new6, y_combine
    #x_resampled7, y_resampled7 = x_new7, y_combine
    #x_resampled8, y_resampled8 = x_new8, y_combine
    #x_resampled9, y_resampled9 = x_new9, y_combine
    #x_resampled10, y_resampled10 = x_new10, y_combine
    #x_resampled11, y_resampled11 = x_new11, y_combine
    ##x_resampled12, y_resampled12 = x_new12, y_combine
    ##x_resampled13, y_resampled13 = rus.fit_sample(x_new13, y_combine)
    #x_resampled_chosen, y_resampled_chosen = x_new_chosen, y_combine
    
    #%applying class-weight
    #----------
    #classifier = SVC(kernel='rbf',  probability=True, class_weight={1: 4.52})
    #classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7, class_weight={1: 4.52} )
    #%% Baseline
    
    #x_resampled1, y_resampled1 = x_combine, y_combine
    #x_resampled2, y_resampled2 = x_combine, y_combine
    #x_resampled3, y_resampled3 = x_combine, y_combine
    #x_resampled4, y_resampled4 = x_combine, y_combine
    #x_resampled5, y_resampled5 = x_combine, y_combine
    #x_resampled6, y_resampled6 = x_combine, y_combine
    #x_resampled7, y_resampled7 = x_combine, y_combine
    #x_resampled8, y_resampled8 = x_combine, y_combine
    #x_resampled9, y_resampled9 = x_combine, y_combine
    #x_resampled10, y_resampled10 = x_combine, y_combine
    #x_resampled11, y_resampled11 = x_combine, y_combine
    ##x_resampled12, y_resampled12 = x_combine, y_combine
    ##x_resampled13, y_resampled13 = rus.fit_sample(x_new13, y_combine)
    #x_resampled_chosen, y_resampled_chosen = x_combine, y_combine
    #
    ##Baseline
    ##---------------
    ##classifier = DummyClassifier(strategy="most_frequent")
    #
    #classifier = SVC(kernel='rbf',  probability=True)
    ##classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7 )
    
    
    
    
    
    #%% Optimize C and gamma in an RBF-Kernel SVM
    #from sklearn.model_selection import GridSearchCV
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-3, 1e-4],
    #                     'C': [1, 10, 100, 1000]},
    #                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    #grid_search.fit(x_resampled1, y_resampled1)
    #predicted = grid_search.predict(test_data_flatten)
    #print('train score:',accuracy_score(train_ground, grid_search.predict(train_data_flatten)))
    #print('test score :',accuracy_score(test_ground, predicted))
    #print('tp:',tp(test_ground, predicted))
    #print('tn:',tn(test_ground, predicted))
    #print('fp:',fp(test_ground, predicted))    
    #print('fn:',fn(test_ground, predicted))
    #   
    #print(grid_search.best_params_)
    
    
    
    
    #%%Leave PGroup Out
    #from sklearn.model_selection import LeavePOut,KFold
    #
    #group_k_fold = KFold(p=20)
    #cv1=list(group_k_fold.split(x_resampled1, y_resampled1,index1))
    #cv2=list(group_k_fold.split(x_resampled2, y_resampled2,index2))
    #cv3=list(group_k_fold.split(x_resampled3, y_resampled3,index3))
    #cv4=list(group_k_fold.split(x_resampled4, y_resampled4,index4))
    #cv5=list(group_k_fold.split(x_resampled5, y_resampled5,index5))
    #cv6=list(group_k_fold.split(x_resampled6, y_resampled6,index6))
    #cv7=list(group_k_fold.split(x_resampled7, y_resampled7,index7))
    #cv8=list(group_k_fold.split(x_resampled8, y_resampled8,index8))
    #cv9=list(group_k_fold.split(x_resampled9, y_resampled9,index9))
    #cv10=list(group_k_fold.split(x_resampled10, y_resampled10,index10))
    #cv11=list(group_k_fold.split(x_resampled11, y_resampled11,index11))
    #cv_chosen=list(group_k_fold.split(x_resampled_chosen, y_resampled_chosen,index_chosen))
    #
    #
    #scores1 = cross_validate(classifier, x_resampled1, y_resampled1,cv = cv1, scoring=scoring,  return_train_score=True)      
    #scores2 = cross_validate(classifier, x_resampled2, y_resampled2,cv = cv2, scoring=scoring,  return_train_score=True)      
    #scores3 = cross_validate(classifier, x_resampled3, y_resampled3,cv = cv3, scoring=scoring,  return_train_score=True)      
    #scores4 = cross_validate(classifier, x_resampled4, y_resampled4,cv = cv4, scoring=scoring,  return_train_score=True)      
    #scores5 = cross_validate(classifier, x_resampled5, y_resampled5,cv = cv5, scoring=scoring,  return_train_score=True)      
    #scores6 = cross_validate(classifier, x_resampled6, y_resampled6,cv = cv6, scoring=scoring,  return_train_score=True)      
    #scores7 = cross_validate(classifier, x_resampled7, y_resampled7,cv = cv7, scoring=scoring,  return_train_score=True)      
    #scores8 = cross_validate(classifier, x_resampled8, y_resampled8,cv = cv8, scoring=scoring,  return_train_score=True)      
    #scores9 = cross_validate(classifier, x_resampled9, y_resampled9,cv = cv9, scoring=scoring,  return_train_score=True)      
    #scores10 = cross_validate(classifier, x_resampled10, y_resampled10,cv = cv10, scoring=scoring,  return_train_score=True)      
    #scores11 = cross_validate(classifier, x_resampled11, y_resampled11,cv = cv11, scoring=scoring,  return_train_score=True)      
    #scores_chosen= cross_validate(classifier, x_resampled_chosen, y_resampled_chosen,cv = cv_chosen, scoring=scoring,  return_train_score=True)       
    #
    
    
    #%%Interpolate CV
    
    #def CV_Interpolate(X,Y,originDSsize):
    ##    X,Y= x_new1,y_combine
    #    from sklearn.metrics import roc_auc_score,precision_score,recall_score
    #    from sklearn.model_selection import StratifiedKFold
    #    
    #    lpo = StratifiedKFold(10)
    #    scores = {}
    #    n=0
    #    for train_indx, test_indx in lpo.split(X,Y):
    #    
    #        n=n+1
    #        print('n=',n)
    #        test_indx_org=[]
    #        for i in range(len(test_indx)):
    #            for j in range(originDSsize):
    #                if np.array_equal(X[test_indx[i],:],X[j,:])==True:
    #                    test_indx_org.append(test_indx[i])
    #        
    ##        plt.subplot(10,2,2*n-1)
    ##        x_sum=np.sum(X[test_indx],axis=1)
    ##        plt.hist(x_sum,10)
    ##        plt.subplot(10,2,2*n)
    ##        x_sum_=np.sum(X[test_indx_org],axis=1)
    ##        plt.hist(x_sum_,10)
    #
    #        #print(Y[test_indx_org])
    #        trainX,trainY, testX,testY= X[train_indx], Y[train_indx],X[test_indx_org], Y[test_indx_org]  
    #        
    #        print ("Train , test size: "+ str(trainY[trainY==0].shape),str(trainY[trainY==1].shape), str(testY[testY==0].shape), str(testY[testY==1].shape))
    #
    #        classifier.fit(trainX, trainY)
    #        preds = classifier.predict(testX)
    #        
    #        if np.sum(testY)!=0:
    #            scores.setdefault('test_AUC', []).append(roc_auc_score(testY,preds))
    #            scores.setdefault('test_Accuracy', []).append(accuracy_score(testY, preds))
    #            scores.setdefault('test_tp', []).append(tp(testY, preds))
    #            scores.setdefault('test_tn', []).append(tn(testY, preds))
    #            scores.setdefault('test_fp', []).append(fp(testY, preds))
    #            scores.setdefault('test_fn', []).append(fn(testY, preds))
    #            scores.setdefault('test_prec', []).append(precision_score(testY, preds))
    #            scores.setdefault('test_rec', []).append(recall_score(testY, preds))
    #            scores.setdefault('train_Accuracy', []).append(accuracy_score(trainY, classifier.predict(trainX)))
    #    for key, value in scores.items():
    #        scores[key] = np.array(value)
    #    return scores
    #scores1 = CV_Interpolate(x_resampled1, y_resampled1, y_combine.shape[0])
    #scores2 = CV_Interpolate(x_resampled2, y_resampled2, y_combine.shape[0])
    #scores3 = CV_Interpolate(x_resampled3, y_resampled3, y_combine.shape[0])
    #scores4 = CV_Interpolate(x_resampled4, y_resampled4, y_combine.shape[0])
    #scores5 = CV_Interpolate(x_resampled5, y_resampled5, y_combine.shape[0])
    #scores6 = CV_Interpolate(x_resampled6, y_resampled6, y_combine.shape[0])
    #scores7 = CV_Interpolate(x_resampled7, y_resampled7, y_combine.shape[0])
    #scores8 = CV_Interpolate(x_resampled8, y_resampled8, y_combine.shape[0])
    #scores9 = CV_Interpolate(x_resampled9, y_resampled9, y_combine.shape[0])
    #scores10 = CV_Interpolate(x_resampled10, y_resampled10, y_combine.shape[0])
    #scores11 = CV_Interpolate(x_resampled11, y_resampled11, y_combine.shape[0])
    #scores_chosen= CV_Interpolate(x_resampled_chosen, y_resampled_chosen, y_combine.shape[0])
    #
    
    
    #%%Evaluation
    tp_ = np.round([scores1['test_tp'].mean(),
           scores2['test_tp'].mean(),
           scores3['test_tp'].mean(),
           scores4['test_tp'].mean(),
           scores5['test_tp'].mean(),
           scores6['test_tp'].mean(),
           scores7['test_tp'].mean(),
           scores8['test_tp'].mean(),
           scores9['test_tp'].mean(),
           scores10['test_tp'].mean(),
           scores11['test_tp'].mean(),
           scores_chosen['test_tp'].mean()],0)
        
    tn_ = np.round([scores1['test_tn'].mean(),
           scores2['test_tn'].mean(),
           scores3['test_tn'].mean(),
           scores4['test_tn'].mean(),
           scores5['test_tn'].mean(),
           scores6['test_tn'].mean(),
           scores7['test_tn'].mean(),
           scores8['test_tn'].mean(),
           scores9['test_tn'].mean(),
           scores10['test_tn'].mean(),
           scores11['test_tn'].mean(),
           scores_chosen['test_tn'].mean()],0)   
        
    fp_ = np.round([scores1['test_fp'].mean(),
           scores2['test_fp'].mean(),
           scores3['test_fp'].mean(),
           scores4['test_fp'].mean(),
           scores5['test_fp'].mean(),
           scores6['test_fp'].mean(),
           scores7['test_fp'].mean(),
           scores8['test_fp'].mean(),
           scores9['test_fp'].mean(),
           scores10['test_fp'].mean(),
           scores11['test_fp'].mean(),
           scores_chosen['test_fp'].mean()],0)
    
    fn_ = np.round([scores1['test_fn'].mean(),
           scores2['test_fn'].mean(),
           scores3['test_fn'].mean(),
           scores4['test_fn'].mean(),
           scores5['test_fn'].mean(),
           scores6['test_fn'].mean(),
           scores7['test_fn'].mean(),
           scores8['test_fn'].mean(),
           scores9['test_fn'].mean(),
           scores10['test_fn'].mean(),
           scores11['test_fn'].mean(),
           scores_chosen['test_fn'].mean()],0)   
    
    
    areaUnderROC = np.round([scores1['test_AUC'].mean(),
           scores2['test_AUC'].mean(),
           scores3['test_AUC'].mean(),
           scores4['test_AUC'].mean(),
           scores5['test_AUC'].mean(),
           scores6['test_AUC'].mean(),
           scores7['test_AUC'].mean(),
           scores8['test_AUC'].mean(),
           scores9['test_AUC'].mean(),
           scores10['test_AUC'].mean(),
           scores11['test_AUC'].mean(),
           scores_chosen['test_AUC'].mean()],2)
    
    auccuracy_train = np.round([scores1['train_Accuracy'].mean(),
           scores2['train_Accuracy'].mean(),
           scores3['train_Accuracy'].mean(),
           scores4['train_Accuracy'].mean(),
           scores5['train_Accuracy'].mean(),
           scores6['train_Accuracy'].mean(),
           scores7['train_Accuracy'].mean(),
           scores8['train_Accuracy'].mean(),
           scores9['train_Accuracy'].mean(),
           scores10['train_Accuracy'].mean(),
           scores11['train_Accuracy'].mean(),
           scores_chosen['train_Accuracy'].mean()],2)
        
    auccuracy_test = np.round([scores1['test_Accuracy'].mean(),
           scores2['test_Accuracy'].mean(),
           scores3['test_Accuracy'].mean(),
           scores4['test_Accuracy'].mean(),
           scores5['test_Accuracy'].mean(),
           scores6['test_Accuracy'].mean(),
           scores7['test_Accuracy'].mean(),
           scores8['test_Accuracy'].mean(),
           scores9['test_Accuracy'].mean(),
           scores10['test_Accuracy'].mean(),
           scores11['test_Accuracy'].mean(),
           scores_chosen['test_Accuracy'].mean()],2)
        
    precision_test = np.round([scores1['test_prec'].mean(),
           scores2['test_prec'].mean(),
           scores3['test_prec'].mean(),
           scores4['test_prec'].mean(),
           scores5['test_prec'].mean(),
           scores6['test_prec'].mean(),
           scores7['test_prec'].mean(),
           scores8['test_prec'].mean(),
           scores9['test_prec'].mean(),
           scores10['test_prec'].mean(),
           scores11['test_prec'].mean(),
           scores_chosen['test_prec'].mean()],2)
        
    recall_test = np.round([scores1['test_rec'].mean(),
           scores2['test_rec'].mean(),
           scores3['test_rec'].mean(),
           scores4['test_rec'].mean(),
           scores5['test_rec'].mean(),
           scores6['test_rec'].mean(),
           scores7['test_rec'].mean(),
           scores8['test_rec'].mean(),
           scores9['test_rec'].mean(),
           scores10['test_rec'].mean(),
           scores11['test_rec'].mean(),
           scores_chosen['test_rec'].mean()],2)
    
    result=np.array([areaUnderROC,auccuracy_test,tp_,tn_,fp_,fn_,precision_test,recall_test,auccuracy_train])
    a_result=np.transpose(result)
    FeatureMethodAUC.append(a_result)
        
    #print (scores)
    #for score, value in scoring.items():
    #    s='test_'+ score
    #    print('\t%s: %0.2f ± %0.2f' % (score, scores[s].mean(), scores[s].std()))
    
    ##find majority element for baseline areaUnderROC
    #l=np.ndarray.tolist(areaUnderROC)
    #print("Max areaUnderROC:", max(set(l)))
    #l=np.ndarray.tolist(auccuracy)
    #print("Max auccuracy:",max(set(l)))#, key = l.count
    
    
    #%%
    import matplotlib.pylab as plt
    x_sum=np.sum(x_new1,axis=1)
    x_mean=np.mean(x_new1,axis=1)
    plt.hist(x_sum,50)
    
    #plt.hist(x_new_chosen[:,0],50)
    
    #plt.scatter(x_new1[:,0],x_sum,c=y_combine)
    #plt.scatter(x_new1[:,0],x_sum)
    
    #%% Classifier with cross validation (Method1,Methd2)
    
    #scores1 = cross_validate(classifier, x_resampled1, y_resampled1,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores2 = cross_validate(classifier, x_resampled2, y_resampled2,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores3 = cross_validate(classifier, x_resampled3, y_resampled3,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores4 = cross_validate(classifier, x_resampled4, y_resampled4,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores5 = cross_validate(classifier, x_resampled5, y_resampled5,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores6 = cross_validate(classifier, x_resampled6, y_resampled6,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores7 = cross_validate(classifier, x_resampled7, y_resampled7,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores8 = cross_validate(classifier, x_resampled8, y_resampled8,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores9 = cross_validate(classifier, x_resampled9, y_resampled9,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores10 = cross_validate(classifier, x_resampled10, y_resampled10,cv = 10, scoring=scoring,  return_train_score=False)      
    #scores11 = cross_validate(classifier, x_resampled11, y_resampled11,cv = 10, scoring=scoring,  return_train_score=False)      
    ##scores12 = cross_validate(classifier, x_resampled12, y_resampled12,cv = 10, scoring=scoring,  return_train_score=False)      
    ##scores13 = cross_validate(classifier, x_resampled13, y_resampled13,cv = 10, scoring=scoring,  return_train_score=False)      
    #
    #scores_chosen= cross_validate(classifier, x_resampled_chosen, y_resampled_chosen,cv = 10, scoring=scoring,  return_train_score=False)       
    
    
    #%% Method3: oversampling within cv on train data
    #No class weight
    #--------------------
    #classifier = SVC(kernel='rbf',  probability=True)
    #classifier = RandomForestClassifier(n_estimators=10, bootstrap = False, min_samples_leaf= 10, max_depth = 7 )
    
    #from sklearn.model_selection import GridSearchCV
    #SVM parameters
    #----------------
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-3, 1e-4],
    #                 'C':  [1e-40,1e-20,1e-16,1e-14,1e-10,1e-08,1e-06,1e-04, 1]},
    #                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #classifier = GridSearchCV(SVC(), tuned_parameters, cv=5)
    
    #Random Forest Regressor
    #---------------
    # Create the parameter grid based on the results of random search 
    #param_grid = {
    #    'bootstrap': [True],
    #    'max_depth': [80, 90, 100, 110],
    #    'max_features': [2, 3],
    #    'min_samples_leaf': [3, 4, 5],
    #    'min_samples_split': [8, 10, 12],
    #    'n_estimators': [100, 200, 300, 1000]
    #}
    ## Create a based model
    #rf = RandomForestRegressor()
    ## Instantiate the grid search model
    #classifier = GridSearchCV(estimator = rf, param_grid = param_grid, 
    #                          cv = 3, n_jobs = -1, verbose = 2)
    
    #Random Froest Classifier
    #---------------------
    #rfc=RandomForestClassifier(random_state=42,n_jobs=-1)
    #param_grid = { 
    #    'n_estimators': [10,30, 50,70],
    #    'max_features': ['auto', 'sqrt', 'log2'],
    #    'max_depth' : [4,5,6,7,8],
    #    'criterion' :['gini', 'entropy']
    #}
    #classifier = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    #
    #
    #def cross_val(xtrain):
    #    from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,average_precision_score, roc_auc_score,precision_score,recall_score
    #    from sklearn.model_selection import StratifiedKFold
    #    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    #    split,k = cv.split(x_combine,y_combine),cv.get_n_splits(x_combine,y_combine)
    #    rus = RandomOverSampler(random_state=42)
    #    
    #    scores = {}
    #    for train, test in split:      
    #        trainX,trainY, testX,testY= xtrain[train], y_combine[train],xtrain[test], y_combine[test]    
    #        
    #        #oversample train set_ No class weight
    #        #-----------------------
    #        x_resampled, y_resampled = rus.fit_sample(trainX,trainY)
    ##       
    #        
    #        # fit a model
    #        classifier.fit(x_resampled, y_resampled)
    #        print(classifier.best_params_) #in case:GridSearchCV
    #        
    #        preds = classifier.predict(testX)
    #        
    #        
    #        scores.setdefault('test_AUC', []).append(roc_auc_score(testY,preds))
    #        scores.setdefault('test_Accuracy', []).append(accuracy_score(testY, preds))
    #        scores.setdefault('test_Percision', []).append(precision_score(testY, preds))
    #        scores.setdefault('test_Recall', []).append(recall_score(testY, preds))
    #        scores.setdefault('test_F1', []).append(f1_score(testY, preds))
    #       
    #        print('--------')
    #        print(scores)
        
    #    for key, value in scores.items():
    #        scores[key] = np.array(value)
    #
    #    return scores.copy()
    #
    #scores1 = cross_val(x_new1)
    #scores2 = cross_val(x_new2)
    #scores3 = cross_val(x_new3)
    #scores4 = cross_val(x_new4)
    #scores5 = cross_val(x_new5)
    #scores6 = cross_val(x_new6)
    #scores7 = cross_val(x_new7)
    #scores8 = cross_val(x_new8)
    #scores9 = cross_val(x_new9)
    #scores10 = cross_val(x_new10)
    #scores11 = cross_val(x_new11)
    #scores_chosen = cross_val(x_new_chosen)
    
    
