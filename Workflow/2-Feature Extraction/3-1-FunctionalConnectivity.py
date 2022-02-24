# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:57:19 2020

@author: mmousa4
"""

#%% Load time series
from nilearn.input_data import NiftiMapsMasker
import nibabel as nib
import numpy as np
import os

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#os.chdir('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\NewResults')

ts = 'aal_'#,'ts_harvard_oxford'#,'ts_dict_learn',,'ts_aal'
#pooled_timeseries= np.load('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\ts_canica.npy')
#final_labels = np.load('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\final_labels.npy')

pooled_timeseries= np.load('adhd_ts_aal.npy')
pooled_timeseries= pooled_timeseries[:,10:-10,:]
#pooled_timeseries= np.load(ts+'spectrumDensity.npy')
#%%
import scipy as sp
from scipy import signal
from nilearn.connectome import GroupSparseCovarianceCV,sym_matrix_to_vec,ConnectivityMeasure

def Periodogram (pooled_timeseries):
    f, pxx_den = sp.signal.periodogram(pooled_timeseries, axis=1)
    np.save(ts+'spectrumDensity.npy',pxx_den)
    
    corr_den=np.zeros((pooled_timeseries.shape[0],pooled_timeseries.shape[2],pooled_timeseries.shape[2]))
    for s in range (0,pxx_den.shape[0]):
        for i in range (0,pooled_timeseries.shape[2]):
            for j in range (0,pooled_timeseries.shape[2]):
                corr_den[s,i,j] = sp.stats.spearmanr(pxx_den[s,:,i],pxx_den[s,:,j])[0]
#                corr_den[s,i,j] = covariance.EmpiricalCovariance(pxx_den[s,:,i],pxx_den[s,:,j])
                
    np.save(ts+'corr_den.npy',corr_den)

def Corr_spearmanr(pooled_timeseries):
    corr_spearmanr=np.zeros((pooled_timeseries.shape[0],pooled_timeseries.shape[2],pooled_timeseries.shape[2]))
    for s in range (0,pooled_timeseries.shape[0]):
        for i in range (0,pooled_timeseries.shape[2]):
            for j in range (0,pooled_timeseries.shape[2]):
                corr_spearmanr[s,i,j] = sp.stats.spearmanr(pooled_timeseries[s,:,i],pooled_timeseries[s,:,j])[0]
                
    np.save(ts+'corr_spearmanr.npy',corr_spearmanr)    

def Corr_kendalltau(pooled_timeseries):
    corr_kendalltau=np.zeros((pooled_timeseries.shape[0],pooled_timeseries.shape[2],pooled_timeseries.shape[2]))
    for s in range (0,pooled_timeseries.shape[0]):
        for i in range (0,pooled_timeseries.shape[2]):
            for j in range (0,pooled_timeseries.shape[2]):
                corr_kendalltau[s,i,j] = sp.stats.kendalltau(pooled_timeseries[s,:,i],pooled_timeseries[s,:,j])[0]
    np.save(ts+'corr_kendalltau.npy',corr_kendalltau)    

    
#from nilearn.connectome import GroupSparseCovarianceCV,sym_matrix_to_vec,ConnectivityMeasure
from nilearn import plotting
from nilearn import datasets

def ConnectivityMeasure_(pooled_timeseries):
    # kind : {"correlation", "partial correlation", "tangent","covariance", "precision"}, optional
    #Correlations-----------
    pooled_timeseries = np.array(pooled_timeseries,dtype='float32')
    connectome_measure = ConnectivityMeasure(kind='correlation')#,vectorize=True,discard_diagonal=True)
    correlations = connectome_measure.fit_transform(pooled_timeseries)
    np.save(ts+'correlation.npy',correlations)
   
    connectome_measure = ConnectivityMeasure(kind='partial correlation')#,vectorize=True,discard_diagonal=True)
    correlations = connectome_measure.fit_transform(pooled_timeseries)
    np.save(ts+'partial_correlation.npy',correlations)
    
    connectome_measure = ConnectivityMeasure(kind='precision')#,vectorize=True,discard_diagonal=True)
    correlations = connectome_measure.fit_transform(pooled_timeseries)
    np.save(ts+'precision.npy',correlations)
    
    connectome_measure = ConnectivityMeasure(kind='tangent')#,vectorize=True,discard_diagonal=True)
    correlations = connectome_measure.fit_transform(pooled_timeseries)
    np.save(ts+'tangent.npy',correlations)
   
    connectome_measure = ConnectivityMeasure(kind='covariance')#,vectorize=True,discard_diagonal=True)
    correlations = connectome_measure.fit_transform(pooled_timeseries)
    np.save(ts+'covariance.npy',correlations)
    

from sklearn import covariance, preprocessing

def GroupSparse(pooled_timeseries):
    
    gl = covariance.EmpiricalCovariance()
    connectivities,cov=[],[]
    for timeseries in pooled_timeseries:
        gl.fit(timeseries)
        gl.precision_.shape
        cov.append(gl.precision_)
   
    np.save(ts+'EmpiricalCovariance.npy',cov)
    
    # Run group-sparse covariance on all subjects
    #----------------------------------------
#    gsc = GroupSparseCovarianceCV(verbose=0)
#    gsc.fit(pooled_timeseries)
#    gsc.covariances_.T.shape
#    np.save(ts+'GSC_covariences.npy',gsc.covariances_.T)
#    
    
    

def GraphicalLasso(pooled_timeseries):
    #https://nilearn.github.io/auto_examples/03_connectivity/plot_simulated_connectome.html#sphx-glr-auto-examples-03-connectivity-plot-simulated-connectome-py
#    myScaler = preprocessing.StandardScaler()
#    X = myScaler.fit_transform(pooled_timeseries)
#    emp_cov = covariance.empirical_covariance(X)
#    pooled_timeseries = covariance.shrunk_covariance(emp_cov, shrinkage=0.8) # Set shrinkage closer to 1 for poorly-conditioned data

    
    # Fit one graph lasso per subject
    #---------------------------------
    try:
        from sklearn.covariance import GraphicalLassoCV
    except ImportError:
        # for Scitkit-Learn < v0.20.0
        from sklearn.covariance import GraphLassoCV as GraphicalLassoCV
    
    gl = GraphicalLassoCV(verbose=0)
    connectivities,cov=[],[]
    for timeseries in pooled_timeseries:
        gl.fit(timeseries)
        gl.covariance_.shape
        cov.append(gl.covariance_)
#        connectivities.append( sym_matrix_to_vec(
#                    gl.covariance_, discard_diagonal=True))
    np.save(ts+'GL_covariences.npy',np.array(cov,dtype='float16') )
    connectivities=np.array(cov,dtype='float16')    
#        ax = plt.subplot(n_displayed, 4, 4 * n + 3)
#        max_precision = gl.precision_.max()
#        plotting.plot_matrix(gl.precision_, axes=ax, vmin=-max_precision,
#                             vmax=max_precision, colorbar=False)
#        if n == 0:
#            plt.title("graph lasso")
#        plt.ylabel("$\\alpha=%.2f$" % gl.alpha_)
    
    # Fit one graph lasso for all subjects at once
    #-------------------------------------------
#    try:
#        from sklearn.covariance import GraphicalLassoCV
#    except ImportError:
#        # for Scitkit-Learn < v0.20.0
#        from sklearn.covariance import GraphLassoCV as GraphicalLassoCV
#    
#    gl = GraphicalLassoCV(verbose=2)
#    gl.fit(np.concatenate(pooled_timeseries))
#    
    
    
#    ax = plt.subplot(n_displayed, 4, 4)
#    max_precision = gl.precision_.max()
#    plotting.plot_matrix(gl.precision_, axes=ax, vmin=-max_precision,
#                         vmax=max_precision, colorbar=False)
#    plt.title("graph lasso, all subjects\n$\\alpha=%.2f$" % gl.alpha_)
#
#    show()
    
    #plot
    #----------------
    #regions related to the pooled_timeseries 
    
#    msdl_atlas_dataset = datasets.fetch_atlas_msdl()
#    atlas_img = msdl_atlas_dataset.maps
#    atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(atlas_img)
#    labels = msdl_atlas_dataset.labels
#    
#    plotting.plot_connectome(gl.covariance_,
#                             atlas_region_coords, edge_threshold='90%',
#                             title="Covariance",
#                             display_mode="lzr")
#    plotting.plot_connectome(-gl.precision_, atlas_region_coords,
#                             edge_threshold='90%',
#                             title="Sparse inverse covariance (GraphicalLasso)",
#                             display_mode="lzr",
#                             edge_vmax=.5, edge_vmin=-.5)
#    plot_matrices(gl.covariance_, gl.precision_, "GraphicalLasso", labels)
#    
#    title = "GroupSparseCovariance"
#    plotting.plot_connectome(-gsc.precisions_[..., 0],
#                             atlas_region_coords, edge_threshold='90%',
#                             title=title,
#                             display_mode="lzr",
#                             edge_vmax=.5, edge_vmin=-.5)
#    plot_matrices(gsc.covariances_[..., 0],
#                  gsc.precisions_[..., 0], title, labels)
#    
#    plotting.show()
    

#%%
ConnectivityMeasure_(pooled_timeseries)
Periodogram (pooled_timeseries)
Corr_spearmanr(pooled_timeseries)
Corr_kendalltau(pooled_timeseries)
GraphicalLasso(pooled_timeseries)
GroupSparse(pooled_timeseries)

