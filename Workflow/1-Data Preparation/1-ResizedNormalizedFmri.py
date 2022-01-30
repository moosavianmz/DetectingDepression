# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:48:18 2020

@author: mmousa4
"""

#%%import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.backend import tensorflow_backend as K, resize_volumes
import matplotlib.pyplot as plt
import nibabel as nib


#%% Loading data
import numpy as np
#HPC
#------------
di = "/ddnB/work/mmousa4/Brain/Greening_Data/"
smripath =  di+ "participants/"
srcpath =  di + "Out/fmriprep/"
srcpath1 = di+ "4D_DataFolder/All/"
labelpath="/ddnB/work/mmousa4/Brain/new_data_on_400/SubjectsInBDIFormat1.xlsx"
confoundPath=di +"confounds_9/"
templatePath = di +'template_img_ordered.nii.gz'

#MY Computer
#------------
di = "C:/Users/mmousa4/Documents/Greening_Data/"
smripath = di+ "participants/"
srcpath =  di +"fmriOutput/fmriprep/"#"C:\\Users\\mmousa4\\Documents\\Greening_Data\\participants\\"
srcpath1 = di+ "4D_DataFolder/All/"
labelpath="C:/Users/mmousa4/Documents/Zhachary/BIDS_Folder_Format/SubjectsInBDIFormat1.xlsx"
confoundPath= di + "fmriOutput/"+ "confounds_9/"
prepath = di+ "Oversample/"
templatePath = di +'template_img_ordered.nii.gz'


#%%Loading fMRI preprocessed data, confounds, and labels 

subjectsList= sorted(os.listdir(srcpath))
np.save('subjectsList.npy',subjectsList)
df_label = pd.read_excel(labelpath, header =0 , index_col=0 )

func_filenames_o,stru_filenames_o=[],[]
confounds =[]
time_series_list=[]
labels=[]
selectedSubjects = []
for participant in subjectsList:    
#    func_file= srcpath+ participant + "/func/"+ participant+ "_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    func_file= srcpath1+ participant +"_res4d.nii.gz"
#    strc_file = smripath + df_label.BDI_Format.get(participant,'**')  +"/V2/T1/T1.nii.gz" 
#    print (strc_file)
    if os.path.exists(func_file) and not np.isnan(df_label.loc[participant,'BDI']) and len(func_filenames_o) < len(subjectsList):#os.path.exists(strc_file) and 
#        stru_filenames.append (strc_file)
        selectedSubjects.append(participant)
        func_filenames_o.append(func_file)
        labels.append(df_label.loc[participant,'BDI'])
        ##---Load existing confounds -
        confound_csv = confoundPath +participant+ "_task-rest_desc-confounds_regressors.csv"
        confounds.append(confound_csv)
func_filenames_o = np.asarray(func_filenames_o)
stru_filenames_o = np.asarray(stru_filenames_o)
confounds= np.asarray(confounds)
dfSelected = df_label.loc[selectedSubjects]
dfSelected.to_csv('selecedSubjects.csv')

print("#fmri subjects:",len(func_filenames_o))
np.save('func_filenames9110991.npy', func_filenames_o)     
np.save('confounds.npy',confounds)
np.save('labels.npy',labels)
np.save('selectedSubjects.npy',selectedSubjects)
#%%Binarizing labels: Threshhold for BDI-II is 14
        
final_labels_o = np.zeros((len(labels)))
for x in range(0, len(labels)):    
    if labels[x]<14:
        final_labels_o[x] = 0    
    else:
        final_labels_o[x] = 1
numberofMDDs = len(final_labels_o[final_labels_o==1])
numberofHCs = len(final_labels_o[final_labels_o==0])
print ("#MDDs:",numberofMDDs," #HCs:",numberofHCs)

#%%DownSampling

# Indicies of each class' observations
#i_class0 = np.where(final_labels_o == 0)[0]
#i_class1 = np.where(final_labels_o == 1)[0]
#
## Number of observations in each class
#n_class0 = len(i_class0)
#n_class1 = len(i_class1)
#
## For every observation of class 0, randomly sample from class 1 without replacement
#i_class0_downsampled = np.random.choice(i_class0, size=n_class1, replace=False)
#
## Join together class 0's target vector with the downsampled class 1's target vector
#final_labels_d = np.hstack((final_labels_o[i_class1],final_labels_o[i_class0_downsampled]))
#func_filenames_d = np.hstack((func_filenames_o[i_class1],func_filenames_o[i_class0_downsampled]))
##print("#Downsampled:", len(func_filenames))

#%% 
final_labels = final_labels_o
func_filenames = func_filenames_o
np.save ('final_labels.npy',final_labels)

#%%
#from nilearn.image import load_img , resample_img
#f_img =[]
#for filename in func_filenames:    
#    data = load_img(filename)
#    f_img.append(data)
#func_filenames= np.asarray(f_img)
#np.save('func_filenames9110991', func_filenames)

#%%
#from nilearn import datasets
#adhd_data=datasets.fetch_adhd(n_subjects=100)
#func_filenames = adhd_data.func
#final_labels=adhd_data.phenotypic
#%%Converting Nifti to array & Resizing arrays to fit in the memory 
#%  Image Resulotion: downsampled from the resolution of 91 × 109 × 91 to 45 × 54 × 45 
#-------------------------
#reference of 45 × 54 × 45: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7949139
from nilearn.image import load_img , resample_img
from scipy.ndimage import zoom
import nibabel as nib
#
#img_width, img_height, img_depth = 91,109,91
#img_width, img_height, img_depth = 64,64,31
img_width, img_height, img_depth = 32,32,32
#img_width, img_height, img_depth = 88,104,88
 # For each image at the index-th time step, do this

#components_img = load_img(func_filenames)
f_img =[]
i=0

for filepath in func_filenames :
    print('1',i)
#    img = nib.load(filepath).get_fdata()
    img = load_img(filepath).dataobj
    new_x=img_width/img.shape[0]
    new_y=img_height/img.shape[1]
    new_z=img_depth/img.shape[2]
    time_length =img.shape[3]



    new_img = []
    for index in range(time_length):
        #z_img = img[:,:,:,index]
        z_img = zoom(img[:,:,:,index], (new_x,new_y,new_z), order=1)
        #[Gray scale]  Normalized (scaled) between 0 and 255
        #z_img=np.interp(z_img, (np.nanmin(z_img), np.nanmax(z_img)), (-255, 255))

        #https://en.wikipedia.org/wiki/Normalization_(statistics)
        #https://en.wikipedia.org/wiki/Feature_scaling
        #z_img=(z_img-np.min(z_img))/(np.max(z_img)-np.min(z_img))
        #z_img[z_img<0]=z_img[z_img<0]/np.min(z_img)       
        #z_img[z_img>0]=z_img[z_img>0]/np.max(z_img)       
#        z_img=2*((z_img-np.min(z_img))/(np.max(z_img)-np.min(z_img)))-1
#        z_img[z_img==0]=10000

        new_img.append(z_img.reshape((img_width, img_height, img_depth )))
        #new_img.append(z_img)
        
#    a = np.moveaxis(np.array(new_img), 0, 3)
    #Change Background value
        
    
    b=nib.Nifti1Image(np.array(new_img), np.eye(4))
    f_img.append(b)
    i=i+1

func_filenames= np.asarray(f_img)
print("Image Rescaled shape:",func_filenames.shape )
#np.save('func_filenames9110991', func_filenames)     

#%% Selecting 64 timesteps and saving the result into memory

#width,height,depth,timesteps = 64,64,31,64
width,height,depth,timesteps = 32,32,32,64
num_ent_img=1 #number of images with top entropy to be chosen
signal_3d_resized_indv=np.zeros((len(func_filenames),width,height,depth,timesteps), dtype=np.float16)
signal_3d_resized_1d_indv=np.zeros((len(func_filenames),width*height*depth*timesteps), dtype=np.float16)
i=0
for filename in func_filenames:
    print('2',i)
    data = load_img(filename).dataobj
    data_resized = data[:,:,:,28:28+64]
    signal_3d_resized_indv[i,:,:,:] = data_resized.copy()
    #%Reshape to 1d signal
    signal_3d_resized_1d_indv[i,:] = np.reshape(data_resized, (-1))
    i=i+1
    np.save('signal_3d_resized_indv_323232.npy', signal_3d_resized_indv)
    np.save('signal_3d_resized_1d_indv_323232.npy', signal_3d_resized_1d_indv)
