# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:44:43 2019

@author: mmousa4
"""

#%% Fetch resting state functional datasets and confounds_Load file names

import os
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.image import load_img
#%%

srcpath =  "C:\\Users\\mmousa4\\Documents\\Greening_Data\\fmriOutput\\fmriprep\\"#"C:\\Users\\mmousa4\\Documents\\Greening_Data\\participants\\"
subjectsList= os.listdir(srcpath)
labelpath="C:\\Users\\mmousa4\\Documents\\Zhachary\\BIDS_Folder_Format\\SubjectsInBDIFormat.xlsx"
df_label = pd.read_excel(labelpath, header =0 , index_col = 2 )
conf="confounds_27\\"
confoundPath="C:\\Users\\mmousa4\\Documents\\Greening_Data\\fmriOutput\\"
func_filenames=[]
confounds_27,confounds_3, confound_all =[],[],[]
time_series_list=[]
labels=[]
for participant in subjectsList :
    func_file= srcpath+ participant + "\\func\\"+ participant+ "_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"# "\\V2\\T1\\T1.nii.gz" #"\\V2\\rsfmri\\rsfmri.nii.gz"
    if os.path.exists(func_file) and not np.isnan(df_label.loc[participant,'BDI']) and len(func_filenames) < len(subjectsList):
        func_filenames.append(func_file)
        labels.append(df_label.loc[participant,'BDI'])
        ##---Load existing confounds -
        confounds_27.append(confoundPath+ "confounds_27\\" +participant+ "_task-rest_desc-confounds_regressors.csv")
        
        confounds_3.append( confoundPath+ "confounds_3\\" +participant+ "_task-rest_desc-confounds_regressors.csv")
        
        confound_all.append(confoundPath+ "confounds_all\\" +participant+ "_task-rest_desc-confounds_regressors.csv")
        

        ##---Make confounds
#        confound_file = srcpath+ participant + "\\func\\"+ participant+ "_task-rest_desc-confounds_regressors.tsv"
#        df = pd.read_csv(confound_file, sep='\t', header=0)
#        df.replace([np.inf, -np.inf], np.nan)
#        df.fillna(0,inplace = True)
#        #df = df.loc[:, 'csf':'global_signal']#df.loc[:, 'csf':'rot_z']#
#        confound_csv = "C:\\Users\\mmousa4\\Documents\\Greening_Data\\fmriOutput\\confounds_all\\"+participant+ "_task-rest_desc-confounds_regressors.csv"
#        df.to_csv(confound_csv )
      
        
#%% Fetching "Labels"
final_labels = np.zeros((len(labels)))
for x in range(0, len(labels)):    
    if labels[x]<14:
        final_labels[x] = 0    
    else:
        final_labels[x] = 1
numberofMDDs = len(final_labels[final_labels==1])
numberofHCs = len(final_labels[final_labels==0])

#%%Templates 
#reference: http://www.brainmap.org/icns/
from nilearn.image import concat_imgs
import os
templatePath="C:\\Users\\mmousa4\\Documents\\Greening_Data\\maps\\maps\\"
mapsList= os.listdir(templatePath)
templates=[]
for i in range (1,21):
    templates.append(templatePath + "thresh_zstat"+str(i)+".nii.gz")
template_img=concat_imgs(templates)
template_img.to_filename('C:\\Users\\mmousa4\\Documents\\Python Scripts\\template_img_ordered.nii.gz')


#templates=[]                 
#for ic in mapsList :
#    if ic.split(".")[-1] == 'gz':
#        templates.append(templatePath+ic)
#x=concat_imgs(templates)
#x.to_filename('C:\\Users\\mmousa4\\Documents\\Python Scripts\\template_img_mix.nii.gz')        
#%%Balancing data: Oversampling

#from nilearn.image import resample_img
#
##def Augmentation(func_file,final_labels,s_augment,numtotalimage): 
#numtotalimage = 100    
#    
#print ('training data size : ' + str (numtotalimage) + ';')# + s_augment )
#n_sample_1=len(final_labels[final_labels==1])
#n_sample_0=len(final_labels[final_labels==0])
#   
#total_sample_1=int(numtotalimage/2)
#total_sample_0=int(numtotalimage/2)
#    
#if total_sample_0>n_sample_0:
#    n_rpt_1=(total_sample_1-n_sample_1)
#    n_rpt_0=(total_sample_0-n_sample_0)
#    train_data_1 = np.asarray(func_filenames)[final_labels==1]
#    train_data_0 = np.asarray(func_filenames)[final_labels==0]
#    
#    indx_rnd_1=np.random.randint(0,n_sample_1,n_rpt_1)
#    indx_rnd_0=np.random.randint(0,n_sample_0,n_rpt_0)
#    
#    train_data_rpt_1=[]
#    #train_data_rpt_1=train_data_1[indx_rnd_1,...].copy()
##    img = func_file[indx_rnd_1]
#    number = 0
#    prepath = "C:\\Users\\mmousa4\\Documents\\Greening_Data\\Oversample\\"
#    
#    for x in train_data_1[indx_rnd_1,...]:
#        names = x.split('\\')[-1]
#        subject = names.split('_')[0]
#        number += 1
#        new_filename=os.path.join(prepath+"1\\" , str(number) +"_"+ names)
#        resample_img(load_img(x)).to_filename(new_filename)
#        func_filenames.append(new_filename)
#        
#        confound_csv = confoundPath+subject+ "_task-rest_desc-confounds_regressors.csv"
#        confounds.append(confound_csv)
##        final_labels.append(1)
#            
#    
#    train_data_rpt_0=[]
##    train_data_rpt_0=train_data_0[indx_rnd_0,...].copy()
#    for x in train_data_0[indx_rnd_0,...]:
#        names = x.split('\\')[-1]
#        subject = names.split('_')[0]
#        number += 1
#        new_filename=os.path.join(prepath+"0\\" , str(number) +"_"+ names)
#        resample_img(load_img(x)).to_filename(new_filename)
#        func_filenames.append(new_filename)
#        confound_csv = confoundPath +subject+ "_task-rest_desc-confounds_regressors.csv"
#        confounds.append(confound_csv)
##        final_labels.append(0)
##train_ground_oversample=np.concatenate((train_ground,[0]*n_rpt_0,[1]*n_rpt_1),axis=0)
#final_labels = np.concatenate((final_labels,[1]*n_rpt_1,[0]*n_rpt_0),axis=0)


#%% CanCIA- No confound
from nilearn.decomposition import CanICA
from nilearn.decomposition import DictLearning
from nilearn.input_data import NiftiMapsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure,sym_matrix_to_vec
from sklearn.externals.joblib import Memory
from nilearn import plotting


#mem = Memory('nilearn_cache')

#CANICA-------------
n_components =30

canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=1,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_filenames)
# Retrieve the components
components = canica.components_
components_img = canica.components_img_


#Plotting ICA Maps------------
plotting.plot_prob_atlas(components_img, view_type='filled_contours',title='ICA maps without confound: ,components:'+str(n_components))

#%% CanCIA-  confounds_3
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
              #  memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_filenames,confounds=confounds_3)

# Retrieve the components
components = canica.components_
components_img = canica.components_img_
components_img.to_filename('canica_Confounds_3.nii.gz')


#Plotting ICA Maps------------
plotting.plot_prob_atlas(components_img, view_type='filled_contours',title='ICA maps with confound_3: '+',components:'+str(n_components))

#%% CanCIA-  confounds_27
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
               # memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_filenames,confounds=confounds_27)

# Retrieve the components
components = canica.components_
components_img = canica.components_img_
components_img.to_filename('canica_Confounds_27.nii.gz')


#Plotting ICA Maps------------
plotting.plot_prob_atlas(components_img, view_type='filled_contours',title='ICA maps with confound_27: '+',components:'+str(n_components))

#%% CanCIA-  confounds_all
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
               # memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_filenames,confounds=confound_all)

# Retrieve the components
components = canica.components_
components_img = canica.components_img_
components_img.to_filename('canica_Confounds_all.nii.gz')


#Plotting ICA Maps------------
plotting.plot_prob_atlas(components_img, view_type='filled_contours',title='ICA maps with confound_all: '+',components:'+str(n_components))



#%%
#Timeseries-----------
masker_ICA = NiftiMapsMasker(components_img, smoothing_fwhm=6,
                         standardize=True, detrend=True,
                         t_r=2.5, low_pass=0.1,
                         high_pass=0.01)

correlations_ICA = []
pooled_timeseries_ICA = []
for filename, confound in zip(func_filenames, confounds):
    hv_confounds = mem.cache(image.high_variance_confounds)(filename)
    
    timeseries_each_subject = masker_ICA.fit_transform(filename,confounds=[hv_confounds,confound] )
    pooled_timeseries_ICA.append(timeseries_each_subject)

    ## call fit_transform from ConnectivityMeasure object
    #correlation = connectome_measure.fit_transform([timeseries_each_subject])
    ## saving each subject correlation to correlations
    #correlations.append(correlation)
    
#Correlations-----------
connectome_measure = ConnectivityMeasure(kind='correlation')
#connectome_measure = ConnectivityMeasure(kind='correlation', vectorize=True )

correlations_ICA = connectome_measure.fit_transform(pooled_timeseries_ICA)
correlations_ICA_mean = connectome_measure.mean_
Features_correlations_ICA = sym_matrix_to_vec(correlations_ICA, discard_diagonal=True)

#Plotting correlations--------------
plotting.plot_matrix(correlations_ICA_mean, vmax=1, vmin=-1, colorbar=True, title="Correlation ICA-mean")

#%% Template matching
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5403936/  , (Smith et al., 2009)
#using Pearson cross-correlation algorithm implemented in FSL (fslcc).
#We applied Fisher's r-to-z transform using a conservative degrees-of-freedom value of 500 (number of independent resolution elements) 
#and converted the resultant z score to a P-value

#fslcc.cc cross-correlate two time series 
#reference: http://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/tutorial_packages/OSX/fsl_501/src/avwutils/fslcc.cc
from scipy.stats.stats import pearsonr
from nilearn.image import iter_img,concat_imgs
from nilearn.image import resample_to_img
from scipy.signal import correlate
from scipy.spatial.distance import cdist
from scipy.spatial import distance


#x=concat_imgs(templates)
x1=resample_to_img(components_img,x )
#x.to_filename('template_img.nii.gz')
x1.to_filename('components_img.nii.gz')
x1=x1.get_data()
dist = cdist(x1,x,'cityblock')
all_ic_cc=[]
for i, cur_img in enumerate(iter_img(components_img)):
    ic_cc=[]
    temp_similarity = []

    for temp in templates: 
        template = load_img(temp)
        x = resample_to_img(cur_img,template ).get_data()
        y =template.get_data()
        #cc = correlate(x, y, mode='same')
        dist = cdist(x,y,'correlation')  
        ic_cc.append(dist)
        temp_similarity.append( )
    all_ic_cc.append(temp_similarity)
  
#masker_template = NiftiMapsMasker(templates, smoothing_fwhm=6,
#                         standardize=True, detrend=True,
#                         t_r=2.5, low_pass=0.1,
#                         high_pass=0.01)      
#
#timeseries_templates = masker_template.fit_transform(templates)
#%% Extracting regions for CanICA
from nilearn.regions import RegionExtractor

regionExt = RegionExtractor(components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor= 'local_regions',
                            standardize=True, min_region_size=1350)
regionExt.fit()

#Timeseries-----------
correlations_ICA_RE = []
pooled_timeseries_ICA_RE = []
for filename, confound in zip(func_filenames, confounds):
    hv_confounds = mem.cache(image.high_variance_confounds)(filename)
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = regionExt.transform(filename, confounds=[hv_confounds,confound])
    pooled_timeseries_ICA_RE.append(timeseries_each_subject)

#Correlations-----------
connectome_measure = ConnectivityMeasure(kind='correlation', discard_diagonal=True)
#connectome_measure = ConnectivityMeasure(kind='correlation', vectorize=True )

correlations_ICA_RE = connectome_measure.fit_transform(pooled_timeseries_ICA_RE)
correlations_ICA_RE_mean = connectome_measure.mean_
Features_correlations_ICA_RE = sym_matrix_to_vec(correlations_ICA_RE)

#Plotting correlations--------------
plotting.plot_matrix(correlations_ICA_RE_mean, vmax=1, vmin=-1, colorbar=True, title="Correlation ICA-mean")

#Plotting Region Maps---------------
# Extracted regions are stored in regions_img_
regions_extracted_img = regionExt.regions_img_
# Each region index is stored in index_
regions_index = regionExt.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, n_components))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)


#correlations--------------
nodecoords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_extracted_img)
mkp=plotting.plot_connectome(correlations_ICA_RE_mean, nodecoords_connectome,
                        edge_threshold='95%', title="Graph of the Extracted Regions ICA")


#%%Classification
#Reference: https://nilearn.github.io/auto_examples/03_connectivity/plot_group_level_connectivity.html
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3)
svc = LinearSVC(random_state=0)

cv_scores = cross_val_score(svc,
                                Features_correlations_ICA,
                                y=final_labels,
                                cv=cv,
                                groups=final_labels,
                                scoring='accuracy',
                                )
print("Regions from ICA: local_regions",cv_scores.mean())

#%% plotting



   
#%%Create Graph
#referenec: https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/09-fcma.html#connectome
#reference: https://brainiak.org/events/ohbm2018/brainiak_sample_tutorials/09-fcma.html#connectome

import networkx as nx
from networkx.algorithms import community

G = nx.Graph()
#, coords_connectome
epoch_corr= correlations_ICA_RE_mean
# What is the (absolute) correlation threshold
threshold = 0.9

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

nx.draw(G)


#local measures
#betweenness centrality, clustering coefficient, characteristic path,
#community structure Newman, community structure Louvain, eccentricity,
#eigenvector centrality, rich club coefficient, subgraph centrality[60], and participation coefficient
glm_betweennessCentrality=nx.betweenness_centrality(G) #betweenness centrality
glm_clusteringCoefficient= nx.clustering(G) #clustering coefficient
glm_communities_generator = community.girvan_newman(G) #community structure Newman

