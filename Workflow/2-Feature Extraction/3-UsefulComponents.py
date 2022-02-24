# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:44:58 2020

@author: mmousa4
"""
#HPC
#------------
di = "/worka/work/mmousa4/Brain/CNN_MRI_backup/Greening_Data/"
smripath =  di+ "participants/"
srcpath =  di + "Out/fmriprep/"
srcpath1 = di+ "4D_DataFolder/All/"
labelpath="/worka/work/mmousa4/Brain/CNN_MRI_backup/new_data_on_400/SubjectsInBDIFormat1.xlsx"
confoundPath=di +"confounds_9/"
prepath = di+ "Oversample/"
templatePath = di +'template_img_ordered.nii.gz'

#MY Computer
#------------
#di = "C:/Users/mmousa4/Documents/Greening_Data/"
#smripath = di+ "participants/"
#srcpath =  di +"fmriOutput/fmriprep/"#"C:\\Users\\mmousa4\\Documents\\Greening_Data\\participants\\"
#srcpath1 = di+ "4D_DataFolder/All/"
#labelpath="C:/Users/mmousa4/Documents/Zhachary/BIDS_Folder_Format/SubjectsInBDIFormat1.xlsx"
#confoundPath= di + "fmriOutput/"+ "confounds_9/"
#prepath = di+ "Oversample/"
#templatePath = di +'template_img_ordered.nii.gz'
#%%
import scipy
from nilearn import image
from collections import Counter 
import numpy as np 

def usefulComponents(components_img):
    #%Identifying noise components among ICA maps (based on Partner-Matching paper)
    
    zThreshold = 0 
#    templatePath="C:\\Users\\mmousa4\\Documents\\Python Scripts\\template_img_ordered.nii.gz"#Greening_Data\\maps\\maps\\"
    #pathcomponents = 'C:\\Users\\mmousa4\\Documents\\Python Scripts\\canica_30_01282020.nii.gz'
    path_templatemaps = templatePath
    concat_templatemaps = image.load_img(path_templatemaps)
    #concat_componentsmaps = load_img(pathcomponents)._data_cache
    concat_componentsmaps = image.resample_to_img(components_img,concat_templatemaps).dataobj
    concat_templatemaps = image.load_img(path_templatemaps)._data_cache
    
    #To measure the spatial similarity between two ICA components within differing families ofcomponents, 
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
            compare_similarity[i,j,:] = scipy.stats.zscore([Sscc,float(Std),Smi])
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
    
    #%Step 2: Identify single-directional matches for each component in family A with all the components in family B.
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
    for i in range (0,3):
        for j in range (len(compare_similarity[:,0,0])):
            compare_similarity_normalized1[j,:,i] = (compare_similarity[j,:,i] - np.mean(compare_similarity[j,:,i]))/np.std(compare_similarity[j,:,i])
    
    
    for i in range(len(concat_componentsmaps[0,0,0,:])) :
        for j in range(len(concat_templatemaps[0,0,0,:])) :
            for k in range (0,3):
                compare_similarity_normalized1[i,j,k] = np.max(compare_similarity_normalized1[:,j,k]) if np.max(compare_similarity_normalized1[:,j,k])==compare_similarity_normalized1[i,j,k] and np.max(compare_similarity_normalized1[:,j,k])>=zThreshold else 0
    
    # Majority rule of three similarity measures
    for j in range(len(concat_templatemaps[0,0,0,:])) :
        reference = 'b'+str(j)
        matched = 'a' + str(Partner_matching(compare_similarity_normalized1 [:,j,:]))
        d_b[reference] = matched
    
    bi_directional = {}
    
    for i in range(len(concat_componentsmaps[0,0,0,:])) :
        reference = 'a'+str(i)
        possibleMatched =  d_a[reference]
        possibleReference = d_b[possibleMatched]
        if possibleReference == reference : 
            bi_directional[reference] = possibleMatched
#    useful_components =  [int(sub[1:]) for sub in list(bi_directional.values())]   
    
    useful_components = [int(sub[1:]) for sub in list(d_a.keys())]   
    return useful_components
    #%% Load Components
import time 

n_components =30
timestr = time.strftime("%m%d%Y") 
#pathcomponents = 'C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\canica_9110991_30_05262020.nii.gz'
pathcomponents = 'canica_9110991_30_05262020.nii.gz'
  
components_img = image.load_img(pathcomponents)
   
#identify the useful group-level ICs
#########################
useful_components =usefulComponents(components_img)

#%% Load func_filenames
from nilearn.input_data import NiftiMapsMasker
import nibabel as nib

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#func_filenames= np.load('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\func_filenames.npy')
func_filenames= np.load('func_filenames.npy')

#%%Reshape

width,height,depth,timesteps = 32,32,32,56
maps_ICA_resized_indv=np.zeros((len(func_filenames),width,height,depth,timesteps), dtype=np.float16)

def Reshape(index,filename):
    print("2",index)
    img_width, img_height, img_depth = 32,32,32
    new_x,new_y,new_z = img_width/91,img_height/109,img_depth/91
    data = image.load_img(filename).dataobj
    data_resized=[]
    for t in range(8,120,2):
        z_img = scipy.ndimage.zoom(data[:,:,:,t], (new_x,new_y,new_z), order=1)
        z_img=2*((z_img-np.min(z_img))/(np.max(z_img)-np.min(z_img)))-1
        data_resized.append(z_img.reshape((img_width, img_height, img_depth )))
    data_resized = np.moveaxis(np.array(data_resized), 0, 3)    
    maps_ICA_resized_indv[index,:,:,:] = data_resized.copy()    
    
    np.save('maps_ICA_resized_indv_323232.npy', maps_ICA_resized_indv)
    #    np.save('signal_3d_resized_1d_indv.npy', signal_3d_resized_1d_indv)


#%% Subject-Specific ICA Components 
#Subject-Specific Maps
##########################
pooled_timeseries_ICA=[]
maps_ICA_T = []
#width,height,depth,timesteps = 32,32,32,64
#width,height,depth,timesteps = 32,32,32,64
#map_timeseries=np.zeros((len(func_filenames),width,height,depth,timesteps), dtype=np.float16)


#A masker for all good components
#---------------------------
components_img_useful = np.array(components_img.dataobj)[:,:,:,useful_components]

#Components' masker
#---------------------------
#masker_ICA = NiftiMapsMasker(nib.Nifti1Image(components_img_useful, np.eye(4)), smoothing_fwhm=6,
#                         standardize=True, detrend=True,
#                         t_r=2.5, low_pass=0.1,
#                         high_pass=0.01,memory='nilearn_cache', memory_level=1)
# 

masker_ICA = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01,memory='nilearn_cache', memory_level=1)
index=0 
for filename in func_filenames:
    timeseries_each_subject = masker_ICA.fit_transform(filename)
    map_timeseries= masker_ICA.inverse_transform(timeseries_each_subject)
    Reshape(index,map_timeseries) #maps_ICA_resized_indv is ready at the end
    index=index+1
    

        
#a masker per components
#---------------------------
#for filename in func_filenames:
##    for filename in zip(func_filenames):
#    for component in useful_components:
#        a=components_img.dataobj[:,:,:,component]
#        a=a[:,:,:,np.newaxis]
#        masker_ICA_c = NiftiMapsMasker(nib.Nifti1Image(a, np.eye(4)), smoothing_fwhm=6,
#                                 standardize=True, detrend=True,
#                                 t_r=2.5, low_pass=0.1,
#                                 high_pass=0.01,memory='nilearn_cache', memory_level=1)
#
#        
#        #hv_confounds = mem.cache(image.high_variance_confounds)(filename)    
#        timeseries_each_subject = masker_ICA_c.fit_transform(filename)#,confounds=[confound] )
#    
#        pooled_timeseries_ICA.append(timeseries_each_subject)
#        map_timeseries= masker_ICA_c.inverse_transform(timeseries_each_subject)
#        maps_ICA_T.append(np.array(map_timeseries))
#        

print ('maps_ICA_T:' , np.array(maps_ICA_resized_indv).shape)
np.save('maps_ICA_T.npy',np.array(maps_ICA_resized_indv))

