# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 08:00:01 2019

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
#%%
#from mpi4py import MPI
#from multiprocessing import Process, cpu_count
# 
#def do_something_useful(rank, shared_process_number):
#    # Do something useful here.
#    print ('Python hybrid, MPI_Process-local_process (not quite a thread): {}-{}'.format(rank, shared_process_number))
# 
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
# 
#print ('Calling Python multiprocessing process from Python MPI rank {}'.format(rank))
# 
## Create shared-ish processes
#shared_processes = []
##for i in range(cpu_count()):
#for i in range(8):
#    p = Process(target=do_something_useful, args=(rank, i))
#    shared_processes.append(p)
# 
## Start processes
#for sp in shared_processes:
#    sp.start()
# 
## Wait for all processes to finish
#for sp in shared_processes:
#    sp.join()
# 
#comm.Barrier()

#os.environ['MKL_NUM_THREADS'] = '20'
#os.environ['GOTO_NUM_THREADS'] = '20'
#os.environ['OMP_NUM_THREADS'] = '20'
#os.environ['openmp'] = 'True'
#%%
## construct the argument parse and parse the arguments
##https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
#import argparse
#ap = argparse.ArgumentParser()
##ap.add_argument("-o", "--output", required=True,
##	help="path to output plot")
#ap.add_argument("-g", "--gpus", type=int, default=1,
#	help="# of GPUs to use for training")
#args = vars(ap.parse_args())
# 
## grab the number of GPUs and store it in a conveience variable
#G = args["gpus"]
#print("[INFO] training with {} GPUs...".format(G))
#from keras.utils.training_utils import multi_gpu_model

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
n_components =30

    
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,                  
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_filenames,confounds=confounds)

# Retrieve the components
components_img = canica.components_img_
#components_img = canica.masker_.inverse_transform(canica.components_) 
import time
timestr = time.strftime("%m%d%Y")  
components_img.to_filename('canica_'+str(n_components)+'_'+timestr+'.nii.gz')
#pathcomponents = 'C:\\Users\\mmousa4\\Documents\\Python Scripts\\canica_30_01282020.nii.gz'

#components_img = load_img(pathcomponents)

##ICA-----------
masker_ICA = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01,memory='nilearn_cache', memory_level=1)


#%%Identifying noise components among ICA maps (based on Partner-Matching paper)
zThreshold = 0 
#templatePath="C:\\Users\\mmousa4\\Documents\\Greening_Data\\maps\\maps\\"
#pathcomponents = 'C:\\Users\\mmousa4\\Documents\\Python Scripts\\canica_30_01282020.nii.gz'
path_templatemaps = templatePath
concat_templatemaps = load_img(path_templatemaps)
#concat_componentsmaps = load_img(pathcomponents)._data_cache
concat_componentsmaps = resample_to_img(components_img,concat_templatemaps).dataobj
concat_templatemaps = load_img(path_templatemaps)._data_cache

###To measure the spatial similarity between two ICA components within differing families ofcomponents, 
#we first convert each component into a Z-score map.
zscoretemplate = scipy.stats.zscore(concat_templatemaps)
zscoremap = scipy.stats.zscore(concat_componentsmaps)


# threshold each Z-score map of the component across the entire 3D volume
zt,zu,zl=2,8,-8
zscoremap = np.where(zscoremap>= zt, zscoremap,0)
zscoremap = np.where(zscoremap>= zu, zu,zscoremap)
zscoremap = np.where(zscoremap<= zl, zl, zscoremap)

zscoretemplate = np.where(zscoretemplate>= zt, zscoretemplate,0)
zscoretemplate = np.where(zscoretemplate>= zu, zu,zscoretemplate)
zscoretemplate = np.where(zscoretemplate<= zl, zl, zscoretemplate)

#Step 1: Compute and construct a similarity matrix for every two component families, A and B
#----------------------------
compare_similarity = np.zeros((len(concat_componentsmaps[0,0,0,:]),len(concat_templatemaps[0,0,0,:]),3))
for i in range(len(concat_componentsmaps[0,0,0,:])) :
    for j in range(len(concat_templatemaps[0,0,0,:])) :
    
        reshaped_zscore_x = zscoremap[:,:,:,i].reshape(-1)
        reshaped_zscore_y = zscoretemplate[:,:,:,j].reshape(-1)


        #Spatial correlation coefficient: We define  based on Pearson’s correlation coefficient
        #reshaping the 3D volume into a 1D
        pearson_p, Sscc = scipy.stats.pearsonr(reshaped_zscore_x, reshaped_zscore_y)

        #the joint distribution
        Puv,_,_ = np.histogram2d(reshaped_zscore_x, reshaped_zscore_y, bins=10, density=True)
        Puv=Puv/np.sum(Puv)
        Pu=np.sum(Puv, axis=1)
        Pv=np.sum(Puv, axis=0)
        Suv=Puv*np.log(Puv/(Pu*Pv))
        
        Std = (np.matmul(reshaped_zscore_x[:,np.newaxis].T,reshaped_zscore_y[:,np.newaxis]))/(np.matmul(reshaped_zscore_x[:,np.newaxis].T,reshaped_zscore_x[:,np.newaxis])+np.matmul(reshaped_zscore_y[:,np.newaxis].T,reshaped_zscore_y[:,np.newaxis])-np.matmul(reshaped_zscore_x[:,np.newaxis].T,reshaped_zscore_y[:,np.newaxis]))
#        Suv=np.zeros(np.shape(Puv))
#        for i in range(len(Puv)) :
#            for j in range(len(Puv)) :
#                Suv[i,j]=Puv[i,j]*np.log(Puv[i,j]/(Pu[i]*Pv[j]))
             
        Smi=np.nansum(Suv)
        compare_similarity[i,j,:] = scipy.stats.zscore([Sscc,Std,Smi])
#        compare_similarity.append( scipy.stats.zscore([Sscc,Std,Smi]))

#%#
compare_similarity_normalized = np.asarray(compare_similarity).copy()
compare_similarity_normalized1 = np.asarray(compare_similarity).copy()
            
def Partner_matching(otherFamily):
   # otherFamily = compare_similarity_normalized [i,:,:]
    nonZeroIndexesofOtherFamily = np.nonzero(otherFamily)
    freqDict = Counter(nonZeroIndexesofOtherFamily[0])
    if freqDict.most_common()[0][1]>1:
        return freqDict.most_common()[0][0] # index of element in family B
    else:
        return np.argmax(otherFamily[nonZeroIndexesofOtherFamily]) # index of element in family B

#%%
#Step 2: Identify single-directional matches for each component in family A with all the components in family B.
#----------------------------
#normalize into Z-scores
for i in range (0,3):
    for j in range (len(compare_similarity[0,:,0])):
        compare_similarity_normalized[:,j,i] = (compare_similarity[:,j,i] - np.mean(compare_similarity[:,j,i]))/np.std(compare_similarity[:,j,i])
   
for i in range(len(concat_componentsmaps[0,0,0,:])) :
    for j in range(len(concat_templatemaps[0,0,0,:])) :
        for k in range (0,3):
            compare_similarity_normalized[i,j,k] = np.max(compare_similarity_normalized[i,:,k]) if np.max(compare_similarity_normalized[i,:,k])==compare_similarity_normalized[i,j,k] and np.max(compare_similarity_normalized1[i,:,k])>=zThreshold else 0
d_a ,d_b= {},{}
# Majority rule of three similarity measures
for i in range(len(concat_componentsmaps[0,0,0,:])) :
    reference = 'a'+str(i)
    matched = 'b' + str(Partner_matching(compare_similarity_normalized [i,:,:]))    
    d_a[reference] = matched
    
#Step 3: Identify single-directional matches for each component in family B with all the components in family A.
#----------------------------
#normalize into Z-scores
#for i in range (0,3):
#    for j in range (len(compare_similarity[:,0,0])):
#        compare_similarity_normalized1[j,:,i] = (compare_similarity[j,:,i] - np.mean(compare_similarity[j,:,i]))/np.std(compare_similarity[j,:,i])
#
#
#for i in range(len(concat_componentsmaps[0,0,0,:])) :
#    for j in range(len(concat_templatemaps[0,0,0,:])) :
#        for k in range (0,3):
#            compare_similarity_normalized1[i,j,k] = np.max(compare_similarity_normalized1[:,j,k]) if np.max(compare_similarity_normalized1[:,j,k])==compare_similarity_normalized1[i,j,k] and np.max(compare_similarity_normalized1[:,j,k])>=zThreshold else 0
#
## Majority rule of three similarity measures
#for j in range(len(concat_templatemaps[0,0,0,:])) :
#    reference = 'b'+str(j)
#    matched = 'a' + str(Partner_matching(compare_similarity_normalized1 [:,j,:]))
#    d_b[reference] = matched
#%%    
#bi_directional = {}
#
#for i in range(len(concat_componentsmaps[0,0,0,:])) :
#    reference = 'a'+str(i)
#    possibleMatched =  d_a[reference]
#    possibleReference = d_b[possibleMatched]
#    if possibleReference == reference : 
#        bi_directional[reference] = possibleMatched
#useful_components =  [int(sub[1:]) for sub in list(bi_directional.values())]   

#useful_components = [int(sub[1:]) for sub in list(d_a.keys())]   
useful_components = [0, 1]#, 4,5,6,7, 8,9,13,14] #Noise Components: 2, 3, 10,11,12,18, 20
print('useful_components:')
print(useful_components)
n_components = len(useful_components)
#%% Correlations of subjects
# Expand adatbase from 4D fmri to number of 3D components per subject 

    #Timeseries-----------
maps_ICA = []
pooled_timeseries_ICA = []
concat_confounds = []
concat_labels = np.zeros((len(final_labels)*n_components,2)) #Contains (label,subjectID)
counter=0
for filename, confound in zip(func_filenames,confounds):
    #hv_confounds = mem.cache(image.high_variance_confounds)(filename)
    
    timeseries_each_subject = masker_ICA.fit_transform(filename,confounds=[confound] )
    pooled_timeseries_ICA.append(timeseries_each_subject)
    
    masked = masker_ICA.maps_img_.dataobj[:,:,:,useful_components]
    
    if len(maps_ICA)==0:
        maps_ICA=masked.copy()
    else:    
        maps_ICA=np.concatenate((maps_ICA,masked), axis=3)
    
    concat_labels[counter*n_components:(counter+1)*n_components]=final_labels[counter],counter
#    if final_labels[counter] ==1:
#        print(final_labels[counter],counter)
    concat_confounds.extend( [confounds[counter]]*n_components)
    counter=counter+1
concat_confounds = np.asarray(concat_confounds)
numberofMDDs_2 = len(concat_labels[concat_labels==1])
numberofHCs_2 = len(concat_labels[concat_labels==0])




#%%Augmentation
from nilearn.image import resample_img
import shutil



#https://mlnotebook.github.io/post/dataaug/
#translation - offset ∈[−5,5] pixels    
def translateit(image, offset, isseg=False):#the offset is given as a length 2 array defining the shift in the y and x directions respectively (dont forget index 0 is which horizontal row 
    order = 0 if isseg == True else 5 #what kind of interpolation we want to perform: order = 0 means to just use the nearest-neighbour pixel intensity and order = 5 means to perform bspline interpolation with order 5

    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')

#scaling - factor ∈[0.9,1.1] i.e. 10% zoom-in or zoom-out
def scaleit(image, factor, isseg=False):#A factor > 1.0 means the image scales-up, and factor < 1.0 scales the image down.  we should provide a factor for each dimension: if we want to keep the same number of layers or slices in our image. we should set last value to 1.0
    order = 0 if isseg == True else 3

    height, width, depth= image.shape
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = depth

    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row     = (height - zheight) // 2
        col     = (width - zwidth) // 2
        layer   = (depth - zdepth) // 2
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = interpolation.zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        layer   = (zdepth - depth) // 2

        newimg = interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')  
        
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        return newimg

    else:
        return image

#rotations - theta ∈[−10.0,10.0] degrees    
def rotateit(image, theta, isseg=False): #It takes a float for the theta argument which specifies the number of degrees of the roation (negative numbers rotate anti-clockwise)
    order = 0 if isseg == True else 5 #Again we need to specify the order of the interpolation on the new lattice. 
        
    return scipy.ndimage.interpolation.rotate(image, float(theta), reshape=False, order=order, mode='nearest')

def flipit(image, axes): #In this case, we can specify a list of 2 boolean values: if each is 1 then both flips are performed. 
    
    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)
    
    return image    

def Augmentation(func_file,final_labels,s_augment,confound_train): 
    func_file,final_labels,s_augment,confound_train = train_X,train_Y ,'duplicate',train_confound
    func_file_total=func_file.copy()
    numtotalimage = 100    
    confounds_train_oversampled=[]    
    #print ('training data augmentated, numHC+numMDD : ' + str (numtotalimage) + ';')# + s_augment )
    n_sample_1=len(final_labels[final_labels==1])
    n_sample_0=len(final_labels[final_labels==0])
    n_rpt_1,n_rpt_0 = 0 ,0  
    total_sample_1=int(numtotalimage/2)
    total_sample_0=int(numtotalimage/2)
    balanced_more =  0  
    
        
    if total_sample_1>n_sample_1 or n_sample_0>n_sample_1:
        n_rpt_1=(total_sample_1-n_sample_1) if balanced_more else (n_sample_0-n_sample_1)
        train_data_1 = func_file[final_labels==1,:,:,:]
        indx_rnd_1=np.random.randint(0,n_sample_1,n_rpt_1)
        
       
        
       
        for x in train_data_1[indx_rnd_1,:,:,:]:
            
            #Duplicate
            #--------------------------
            new_file = x.copy();
            
             ##Translate,Scale,Rotate Or Flip
              ##----------------------------
#            random_Augmentation = np.random.randint(0,3)
#            if random_Augmentation == 0:
#                 new_file = translateit(x, [0,4], isseg=True)
#            elif random_Augmentation == 1:
#                 new_file = scaleit(x, 0.9, isseg=True)
#            elif random_Augmentation == 2:
#                 new_file = rotateit(x, 5, isseg=True)
#            elif random_Augmentation == 3:
#                new_file = flipit (x, [1,0])
            

            func_file_total = np.concatenate((func_file_total, new_file[np.newaxis,:,:,:]), axis=0)
            
        confounds_train_oversampled.append(confound_train[indx_rnd_1,...])
                
    if total_sample_0>n_sample_0 and balanced_more:        
        n_rpt_0=(total_sample_0-n_sample_0)        
        train_data_0 =func_file[final_labels==0,:,:,:]        
        indx_rnd_0=np.random.randint(0,n_sample_0,n_rpt_0)    
     
        for x in train_data_0[indx_rnd_0,:,:,:]:
             #Duplicate
            #--------------------------
            new_file = x.copy();
             
             
             ##Translate,Scale,Rotate Or Flip
              ##----------------------------
#            random_Augmentation = np.random.randint(0,3)
#            if random_Augmentation == 0:
#                 new_file = translateit(x, [0,4], isseg=True)
#            elif random_Augmentation == 1:
#                 new_file = scaleit(x, 0.9, isseg=True)
#            elif random_Augmentation == 2:
#                 new_file = rotateit(x, 5, isseg=True)
#            elif random_Augmentation == 3:
#                new_file = flipit (x, [1,0])

            func_file_total = np.concatenate((func_file_total, new_file[np.newaxis,:,:,:]), axis=0)
          
        confounds_train_oversampled.append(confound_train[indx_rnd_0,...])
 
    confound_train = np.append(confound_train,np.asarray(confounds_train_oversampled))
    final_labels = np.concatenate((final_labels,[1]*n_rpt_1,[0]*n_rpt_0),axis=0)
    return func_file_total, final_labels,confound_train

#%%
def Vgg16_Seq (train_X,train_Y,test_X,test_Y):
    
#    train_X,test_X, train_Y, test_Y = train_test_split(maps_ICA.T,concat_labels[:,0], test_size=0.2, random_state=42)
#    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(maps_ICA.T, groups=concat_labels[:,1]))
#    train_X, train_Y, test_X, test_Y =maps_ICA.T[train_inds],concat_labels[train_inds,0],maps_ICA.T[test_inds],concat_labels[test_inds,0]

    img_count,img_rows, img_cols, img_depth=train_X.shape[0],train_X.shape[1],train_X.shape[2],train_X.shape[3]
    num_classes=1
    img_input= Input(shape=(img_rows, img_cols, img_depth, 1))
   
    
       # Block 1
    x = layers.Conv3D(64, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv3D(64, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv3D(128, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(128, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block5_pool')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.7)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.7)(x)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
  
    # Create model.
    model = models.Model(img_input, x, name='vgg16')
    
#    model1 = VGG16(weights='imagenet',include_top=True)
##     model1 = VGG_16('vgg16_weights.h5')
#    model.set_weights(model1.get_weights()) 
#    model.load_weights(weights_path)
    
#    #    %% MultiGPU
    
#    if G > 1:
#        # we'll store a copy of the model on *every* GPU and then combine
#    	# the results from the gradient updates on the CPU
#    	#with tf.device("/cpu:0"):
#    		# initialize the model
#        # make the model parallel
#    	model = multi_gpu_model(model, gpus=G)
   # #%%
    tr = np.reshape(train_X,train_X.shape + (1,))
    te = np.reshape(test_X,test_X.shape + (1,))
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)   
    print (model.summary())
    model.compile(optimizer=Adam(lr=00.1), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(tr, train_Y, batch_size=16,epochs=5,verbose=1,validation_data=(te, test_Y),callbacks=[es])
    
    train_features = model.predict(tr)
    test_features= model.predict(te)
        
    train_data_flatten = np.reshape(train_features, (train_features.shape[0], -1))
    test_data_flatten = np.reshape(test_features, (test_features.shape[0], -1))
    
    np.save('train_data_flatten.npy', train_data_flatten)
    np.save('test_data_flatten.npy', test_data_flatten)
#    d = np.load('test3.npy')
    return train_data_flatten, test_data_flatten

    
#Vgg16_Seq ()
#%%
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,0]
#%%
#from scipy import ndimage
#rotate_face = ndimage.rotate(face, 45)
#%% #KFOLD
########################

maps_ICA_T= maps_ICA.T
y =concat_labels[:,0]
cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
print ("StratifiedKFold:",cv)
classifier = SVC(kernel='rbf')
scores,scores1 = {},{}
for train_index, test_index in cv.split(maps_ICA_T,y,groups=concat_labels[:,1]):
    print("**")
    train_X,test_X,train_Y,test_Y  = maps_ICA_T[train_index], maps_ICA_T[test_index], y[train_index], y[test_index]
    train_confound,test_confound = concat_confounds[train_index],concat_confounds[test_index]
    numHC = len(train_Y [train_Y ==0])
    numMDD = len(train_Y [train_Y ==1])
    print ("Train numHC,numMDD: ", numHC,",", numMDD)
    numHC = len(test_Y [test_Y ==0])
    numMDD = len(test_Y [test_Y ==1])
    print ("Test numHC,numMDD: ", numHC,",", numMDD)
    
    
    #Augmentation
    #########################
    
    train_X_a,train_Y_a,train_confound = Augmentation(train_X,train_Y ,'duplicate',train_confound)   
    numHC = len(train_Y_a [train_Y_a ==0])
    numMDD = len(train_Y_a [train_Y_a ==1])
    print ("Augmented Train numHC,numMDD: ", numHC,",", numMDD)
    
    
    #Feature Extractor
    #########################     
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    device_count={"CPU": 80}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 4,   
                intra_op_parallelism_threads = 80)) as sess:
        K.set_session(sess)
        train_data_features,test_data_features = Vgg16_Seq(train_X_a,train_Y_a,test_X,test_Y)
        
           
#    train_data_features,test_data_features = Vgg16_Seq(train_X,train_Y,test_X,test_Y)
    print ("#Feature Extracted : TrainData shape", str(train_data_features.shape)) 
    print ("#Feature Extracted : TestData shape", str(test_data_features.shape))   
    
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