# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:52:08 2020

@author: mmousa4
"""
import numpy as np
import os

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
os.chdir('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\MDD_Data')

func_filenames= np.load('func_filenames9110991.npy')#'adhd_func_filenames.npy')
confounds = np.load('confounds.npy')#'adhd_confounds.npy')
print('func_filename.shape:',func_filenames.shape)
print('confounds.shape:',confounds.shape)

#%%
# find the longest image
def PaddedTimeseries(all_subjects_data):
    max_len_image=np.max([len(i) for i in all_subjects_data])
    
    # reshape 
    
    all_subjects_data_reshaped=[]
    for subject_data in all_subjects_data:
      # Padding
      N= max_len_image-len(subject_data)
      padded_array=np.pad(subject_data, ((0, N), (0,0)), 
                          'constant', constant_values=(0))
      subject_data=padded_array
      subject_data=np.array(subject_data)
      subject_data.reshape(subject_data.shape[0],subject_data.shape[1],1)
      all_subjects_data_reshaped.append(subject_data)
    
    # shape of data
    
    # 40 subjects
    # 261 time stamps
    # 10 netwroks values
    
    print(np.array(all_subjects_data_reshaped).shape)
    return np.array(all_subjects_data_reshaped)


#%%Brain activity maps


    
from nilearn.decomposition import CanICA,DictLearning
from nilearn.input_data import NiftiMapsMasker
from nilearn.regions import RegionExtractor
import nibabel as nib
from nilearn import masking,image
import time
    
def ICA(func_filenames,confounds):
    #CANICA [Group ICA]-------------
    ########################
    n_components =30
#    mask_full = nib.Nifti1Image(np.expand_dims(np.ones(atlas_gt.shape, dtype=np.int8), axis=0), affine=np.eye(4))    
#    mask_img =masking.compute_background_mask(func_filenames)
#    mask_strategy: {‘background’
    
    
    canica = CanICA(n_components=n_components, smoothing_fwhm=6., 
                    threshold=3., verbose=1, random_state=0 ,mask_strategy='template')
    canica.fit(func_filenames,confounds=confounds)
##    # Retrieve the components
    components_img = canica.components_img_
#
    timestr = time.strftime("%m%d%Y") 
    components_img.to_filename('canica_'+str(n_components)+'_'+timestr+'.nii.gz')
#    path='C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\MDD_Data\\'
#    components_img = image.load_img(path+"canica_30_07062020.nii.gz")
#    fmri_reduced_canica = canica.transform(func_filenames)
#    fmri_reduced_canica.to_filename('canicaReduced_9110991_'+str(n_components)+'_'+timestr+'.nii.gz')

    masker = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01,memory='nilearn_cache', memory_level=1)

    pooled_timeseries= []  
    maps_ica=[]
    for filename, confound in zip(func_filenames, confounds):
        
         timeseries_each_subject = masker.fit_transform(filename,confounds=[confound] )
         pooled_timeseries.append(timeseries_each_subject)
#         maps_ica.append(masker.inverse_transform(timeseries_each_subject))
         
    pooled_timeseries =PaddedTimeseries(pooled_timeseries)
    maps_ica = np.asarray(maps_ica)
    np.save('ts_canica.npy',np.array(pooled_timeseries,dtype='float16')) 
    np.save('maps_canica.npy',np.array(maps_ica,dtype='float16')) 
#%%
    ###DICLEARN--------------

def DICLEARN(func_filenames,confounds):
    #CANICA [Group ICA]-------------
    ########################
    n_components =30
    # Initialize DictLearning object
    dict_learn = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                              memory="nilearn_cache", memory_level=2,
                              random_state=0,mask_strategy='template')
    # Fit to the data
    dict_learn.fit(func_filenames)#,confounds=confounds)
    components_img = dict_learn.components_img_
    
    timestr = time.strftime("%m%d%Y") 
    components_img.to_filename('diclearn_9110991_'+str(n_components)+'_'+timestr+'.nii.gz')
#    fmri_reduced_diclearn = dict_learn.transform(func_filenames)
#    fmri_reduced_diclearn.to_filename('diclearnReduced_9110991_'+str(n_components)+'_'+timestr+'.nii.gz')
    
    
    masker = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01,memory='nilearn_cache', memory_level=1)

    pooled_timeseries= []    
    for filename, confound in zip(func_filenames, confounds):

        
         timeseries_each_subject = masker.fit_transform(filename,confounds=[confound] )
         pooled_timeseries.append(timeseries_each_subject)
    pooled_timeseries =PaddedTimeseries(pooled_timeseries)
    np.save('adhd_ts_dict_learn.npy',np.array(pooled_timeseries,dtype='float16'))

    
#%% # Agglomerative Clustering: ward
#reference:https://nilearn.github.io/auto_examples/03_connectivity/plot_data_driven_parcellations.html#sphx-glr-auto-examples-03-connectivity-plot-data-driven-parcellations-py
from nilearn.regions import Parcellations
def Clustering(func_filenames):
    # We build parameters of our own for this object. Parameters related to
    # masking, caching and defining number of clusters and specific parcellations
    # method.
    
    
    ward = Parcellations(method='ward', n_parcels=1000,
                         standardize=False, smoothing_fwhm=2.,
                         memory='nilearn_cache', memory_level=1,
                         verbose=1,mask_strategy='template')
#     Call fit on functional dataset: single subject (less samples).
    ward.fit(func_filenames)
    ward_labels_img = ward.labels_img_

    # Now, ward_labels_img are Nifti1Image object, it can be saved to file
    # with the following code:
    ward_labels_img.to_filename('ward_parcellation.nii.gz')
#     A reduced dataset can be created by taking the parcel-level average:
#     Note that Parcellation objects with any method have the opportunity to
#     use a `transform` call that modifies input features. Here it reduces their
#     dimension. Note that we `fit` before calling a `transform` so that average
#     signals can be created on the brain parcellations with fit call.
    fmri_reduced_clustering = ward.transform(func_filenames)
    
#    fmri_reduced_clustering.to_filename('ts_clusteringReduced_9110991_'+'_'+timestr+'.nii.gz')
    np.save('ts_wardClustering.npy',np.array(fmri_reduced_clustering,dtype='float16'))
#     Display the corresponding data compressed using the parcellation using
#     parcels=2000.
#    fmri_compressed = ward.inverse_transform(fmri_reduced)
    
    
#%%sparse inverse covariance estimator
#https://nilearn.github.io/connectivity/connectome_extraction.html
from sklearn.covariance import GraphicalLassoCV
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

def ts_harvard_oxford(func_filenames):
    
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_filename = dataset.maps
    labels = dataset.labels
    
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
      
    
    pooled_timeseries= []    
    for filename in func_filenames:
        
         timeseries_each_subject = masker.fit_transform(filename)#,confounds=[confound] )
         pooled_timeseries.append(timeseries_each_subject)
    
    np.save('ts_harvard_oxford.npy',np.array(pooled_timeseries,dtype='float16'))
    
def AAL_atlas(func_filenames,confounds):
    
    dataset = datasets.fetch_atlas_aal()
    atlas_filename = dataset.maps
        
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
      
    
    pooled_timeseries= []    
    for filename,confound in zip(func_filenames,confounds):
        
         timeseries_each_subject = masker.fit_transform(filename,confounds=[confound] )
         pooled_timeseries.append(timeseries_each_subject)
    pooled_timeseries =PaddedTimeseries(pooled_timeseries)
    np.save('ts_aal.npy',np.array(pooled_timeseries,dtype='float16'))
#    atlas_img = nib.load(atlas['maps'])

def Smith_Atlas(func_filenames):    
    smith_atlas = datasets.fetch_atlas_smith_2009()
    smith_atlas_rs_networks = smith_atlas.rsn70

    masker = NiftiMapsMasker(maps_img=smith_atlas_rs_networks,  # Smith atals
                         standardize=True, # centers and norms the time-series
                         memory='nilearn_cache', # cache
                         verbose=0) #do not print verbose
    pooled_timeseries=[]
    
    n=0
    for func_file, confound_file in zip(func_filenames, confounds):
    #    print('***********')
    #    print("func_file:",func_file)
    #    print("confound:",confound_file)
        time_series = masker.fit_transform(func_file, confounds=confound_file)
    #    print(time_series.shape)
        pooled_timeseries.append(time_series)
        
        print(n)
        n=n+1
    pooled_timeseries =PaddedTimeseries(pooled_timeseries)
    np.save('ts_smith.npy',np.array(pooled_timeseries,dtype='float16'))
 #%%
AAL_atlas(func_filenames,confounds)
ICA (func_filenames,confounds)
DICLEARN(func_filenames,confounds)
Clustering(func_filenames)
Smith_Atlas(func_filenames)
#ts_harvard_oxford(func_filenames)
#%% Analysis