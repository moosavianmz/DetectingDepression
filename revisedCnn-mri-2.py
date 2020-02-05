# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:36:08 2019

@author: moosa
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 08:17:43 2019

@author: moosa
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from time import time
import tensorflow as tf
from nilearn import image

#prepath= "C:/Users/moosa/Documents/CNN_MRI_backup"
prepath= "/work/mmousa4/Brain"


#Load dataset
#-------------------
import os
import numpy as np
#Load labels
import pandas as pd
numImages = 300
numtotalimage=7000 #` = total (depressed + non_depressed) images after oversampling
filepath = prepath+"/new_data_on_400/"
patientsInfo = filepath+ '8100_BDI-II_20151124.xlsx'
df_patientInfo = pd.read_excel(patientsInfo, header =1 , index_col = 0 )
depressed_labels=df_patientInfo["DBDI_22"]   

subjectsInPatientFile = df_patientInfo.index
listofsubjectsInPatientFile = subjectsInPatientFile.tolist()
uniquesubjectsInPatientFile = list(set(listofsubjectsInPatientFile))

path =  prepath+"/Greening_Data/participants/"
subjectsList= os.listdir(path)
uniquesubjectsList = list(set(subjectsList))

loadedImages, loadedLabels = [],[]
repeatedSubjects,NoBDIsSubjects = [],[]
#common = list(set(uniquesubjectsInPatientFile) & set(uniquesubjectsList))

for subject in subjectsList:   
    imagePath=path + subject +"/V2/T1/T1.nii.gz"
    if os.path.exists(imagePath):
        load_img= image.load_img(imagePath)
        if subject in subjectsInPatientFile :
            if np.size(depressed_labels[subject]) ==1 and len(loadedLabels)<=numImages:
                loadedLabels.append(depressed_labels[subject])
                loadedImages.append(load_img)
#                print(len(loadedImages))
                if len(loadedImages)==numImages:
                    break
            else:
                repeatedSubjects.append(subject)
            #'A00008326','A00028185','A00033747'
        else:
            NoBDIsSubjects.append(subject)
del df_patientInfo,image, imagePath,prepath,subject, load_img,filepath,patientsInfo,depressed_labels,subjectsInPatientFile,listofsubjectsInPatientFile,uniquesubjectsInPatientFile,path,uniquesubjectsList,subjectsList,repeatedSubjects,NoBDIsSubjects
#%%  #Load data and labels
print("Labels")
y_train=np.zeros((len(loadedLabels[0:numImages])))

for x in range(0, len(loadedLabels[0:numImages])):
#    print(loadedLabels[x])
    if loadedLabels[x]<14:
        y_train[x] = 0    
    else:
        y_train[x] =1
        
label_initial=y_train.copy()
del x,loadedLabels,y_train
#%% #Resize images

import skimage
from skimage.transform import resize
from skimage.color import rgb2gray
#numslice=10    

print('image resize starts')

loadedImages_np=[]
image_new_size=(96,96,100) #nilearn template.shape is (91,109,91)
for item in loadedImages:
    item_np=np.array(item.dataobj) #Convert to numpy
    
    item_np_resized=skimage.transform.resize(item_np,image_new_size,mode='constant',cval=0)
    loadedImages_np.append(item_np_resized)

loadedImages_resized=np.array(loadedImages_np)

print('image resize ends')

#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.imshow(loadedImages_resized[100,:,:,50])

del item_np,item_np_resized,image_new_size, item, loadedImages_np
#%% #Choose images based on their entropy
import matplotlib.pyplot as plt

CalcEntropy='yes'
num_ent_img=10 #number of images with top entropy to be chosen
#ent_img = (numer of images , number of slices)
if CalcEntropy=='yes':
    ent_img=np.zeros((loadedImages_resized.shape[0],loadedImages_resized.shape[3])) #Initilize entropy (#images,#slices)
    for i in range (0,loadedImages_resized.shape[0]):
        for j in range (0,loadedImages_resized.shape[3]):
            hist,bins=np.histogram(loadedImages_resized[i,:,:,j])
            prob=hist/np.sum(hist) #Probability
            ent=-np.sum(prob[prob!=0]*np.log2(prob[prob!=0])) #Entropy
            ent_img[i,j]=ent
#            plt.hist(x=loadedImages_resized[i,:,:,j],bins=bins)

    ent_sort_indx=np.argsort(ent_img,axis=1) #index of sorted array
    
    loadedImages_initial=np.zeros((loadedImages_resized.shape[0],loadedImages_resized.shape[1],loadedImages_resized.shape[2],num_ent_img))
    for i in range (0,loadedImages_resized.shape[0]):
        for j in range (0,num_ent_img,1):
            loadedImages_initial[i,:,:,j]=loadedImages_resized[i,:,:,ent_sort_indx[i,j-num_ent_img]].copy()
    
else:
    loadedImages_initial=loadedImages_resized.copy()

#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.imshow(loadedImages_initial[100,:,:,9])
    
del loadedImages_resized,num_ent_img, prob, hist,i,j,CalcEntropy, bins, ent, ent_img, ent_sort_indx
#%% #Split a 3d image to a series of 2d images 
#Each slice in 3d image consider as on 2d image
#Split an iamge of (L1,L2,L3,L4) to (L1*L4,L2,L3,1)

SplitImages='yes'
if SplitImages=='yes':
    loadedImages_gray=np.zeros((loadedImages_initial.shape[0]*loadedImages_initial.shape[3],loadedImages_initial.shape[1],loadedImages_initial.shape[2],1)) #Initilize entropy
    label_gray=np.zeros(loadedImages_initial.shape[0]*loadedImages_initial.shape[3]) #Initilize entropy
    n=0 #Counter
    for i in range (0,loadedImages_initial.shape[0]):
        for j in range (0,loadedImages_initial.shape[3]):
            loadedImages_gray[n,:,:,0]=loadedImages_initial[i,:,:,j].copy()
            label_gray[n]=label_initial[i].copy()
            n=n+1

else:
    loadedImages_gray=loadedImages_initial.copy()
    label_gray=label_initial.copy()
del n,SplitImages,i,j,label_initial,loadedImages_initial
#%% #Convert grayscale image to RGB image
#iamges of (L1,L2,L3,1) convert to (L1,L2,L3,3)

from skimage.color import gray2rgb
def grayToRBG (loadedImages_gray):
    Covert2RGB='yes'
    if Covert2RGB=='yes':
        loadedImages_RGB=np.zeros((loadedImages_gray.shape[0],loadedImages_gray.shape[1],loadedImages_gray.shape[2],3)) #Initilize entropy
       
        for i in range (0,loadedImages_gray.shape[0]):
            img_RGB=gray2rgb(loadedImages_gray[i,:,:,0])
            loadedImages_RGB[i,:,:,:]=img_RGB.copy()
    
    else:
        loadedImages_RGB=loadedImages_gray.copy()
       
    
    return loadedImages_RGB
label_RGB=label_gray.copy()
loadedImages_RGB= grayToRBG (loadedImages_gray)
loadedImages_final=loadedImages_RGB.copy()
label_final=label_RGB.copy()
numHC = len(label_final [label_final ==0])
numMDD = len(label_final [label_final ==1])
print ("numHC =",numHC)
print ("numMDD =",numMDD)
    
del loadedImages_gray,loadedImages_RGB,label_gray,label_RGB
#%% Defining Models : 

#CNN architecture: The architecture consists of fully-connected (FC) and convolutional (Conv) layers and is the following: FC1 -> tahn activation -> FC2 -> tanh activation -> Conv1 -> ReLU activation -> Conv2 -> ReLU activation -> de-Conv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,concatenate
from keras.layers.convolutional import Conv2D,Conv3D
from keras.layers.pooling import MaxPooling2D,MaxPooling3D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from keras import utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import skimage 

#Classifier

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

epochs=2
batch_size=12

    
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix, f1_score, roc_auc_score,precision_score,recall_score
from scipy import ndarray
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,0]

def Normalization (data):
    copy_data = data.copy()
    for i in range (0,np.shape(copy_data)[0]):
        copy_data[i]=(copy_data[i] - copy_data[i].mean()) / copy_data[i].std()
    copy_data=np.reshape(copy_data,copy_data.shape + (1,))
    return copy_data

def random_rotation(image_array:ndarray):
#    reference:https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec

    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = np.random.uniform(-45, 45)
    #print(str(random_degree), " degree" )
    img= skimage.transform.rotate(image_array, random_degree)
    img = gray2rgb(img)
    img=np.reshape(img,img.shape + (1,))
    return img

       
def Augmentation(train_data,train_ground,s_augment): 
    
    print ('training data size : ' + str (numtotalimage) + ';' + s_augment )
    n_sample_1=len(train_ground[train_ground==1])
    n_sample_0=len(train_ground[train_ground==0])
    train_data_1 = train_data[train_ground==1]
    train_data_0 = train_data[train_ground==0]
    
    total_sample_1=int(numtotalimage/2)
    total_sample_0=int(numtotalimage/2)
        
    if total_sample_0>n_sample_0:
        n_rpt_1=(total_sample_1-n_sample_1)
        n_rpt_0=(total_sample_0-n_sample_0)
        
        indx_rnd_1=np.random.randint(0,n_sample_1,n_rpt_1)
        indx_rnd_0=np.random.randint(0,n_sample_0,n_rpt_0)
        
        train_data_rpt_1=[]
        if s_augment=='duplicate':
            train_data_rpt_1=train_data_1[indx_rnd_1,...].copy()
        elif s_augment=='rotate':
            for i in range (0,len(indx_rnd_1)):
                ing = random_rotation(train_data_1[indx_rnd_1[i],:,:,0,0])
                
                train_data_rpt_1.append(ing)
        
                
        
        train_data_rpt_0=[]
        if s_augment=='duplicate':
            train_data_rpt_0=train_data_0[indx_rnd_0,...].copy()
        elif s_augment=='rotate':
            for i in range (0,len(indx_rnd_0)):
                train_data_rpt_0.append(random_rotation(train_data_0[indx_rnd_0[i],:,:,0,0]))
        
        train_data_oversample=np.concatenate((train_data,train_data_rpt_0,train_data_rpt_1),axis=0)
        train_ground_oversample=np.concatenate((train_ground,[0]*n_rpt_0,[1]*n_rpt_1),axis=0)
        
#        print (len(train_ground_oversample[train_ground_oversample==1]))
#        print (len(train_ground_oversample[train_ground_oversample==0]))
        
        train_data=train_data_oversample.copy()
        train_ground=train_ground_oversample.copy()
    
        return train_data,train_ground


#%%
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras import layers
from keras import models
from sklearn.metrics import log_loss
from keras.applications import VGG16,inception_v3



    #%%
def FineTuned_Pretrained3( b_FC):
    #b_FC =1
    
    shape0,shape1,shape2,shape3=loadedImages_final.shape[0],loadedImages_final.shape[1],loadedImages_final.shape[2],loadedImages_final.shape[3]
    model1 = VGG16(weights='imagenet',include_top=False, input_shape=(shape1,shape2,shape3))
    w_model1=list(model1.get_weights())
    x = Flatten(name='flatten')(model1.output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid')(x)
    model2 = models.Model(input=model1.input,output=x ) 
    w_model2=list(model2.get_weights())
#    model2.set_weights(model1.get_weights()) 
    if (not(b_FC)):
        print("NotTrainable: [:-6]")

        for layer in model2.layers[:-6]:
            layer.trainable = False
    else:
        print("NotTrainable: [:-4]")
        for layer in model2.layers[:-4]:
            layer.trainable = False

    print("Model2.Summary:")
    model2.summary()
    model2.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
    #Extract features from the convolutional base  
    train_X = np.reshape(loadedImages_final, (shape0,shape1,shape2,shape3 ))
    
    model2.fit(train_X, label_final, batch_size=batch_size,epochs=epochs,verbose=1)
    w_model2_fit=list(model2.get_weights())
    #model2.save('model2_complete.h5')
    model3 = models.Model(input=model2.input,output=model2.layers[-3].output)
    print("Model3.Summary:")
    model3.summary()

    print("\n")
    train_features = model3.predict(train_X)
    
    train_data_flatten = np.reshape(train_features, (train_features.shape[0], -1))
    return train_data_flatten

    #%%
def FineTuned_Pretrained3_Inception(b_FC):
    
    shape0,shape1,shape2,shape3=loadedImages_final.shape[0],loadedImages_final.shape[1],loadedImages_final.shape[2],loadedImages_final.shape[3]
    model1 = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(shape1,shape2,shape3))
#    w_model1=list(model1.get_weights())
  
    x = layers.GlobalAveragePooling2D()(model1.output)
    # let's add a fully-connected layer
#    x = Dense(512, activation='relu')(x)        
    x = Dense(1, activation='sigmoid')(x)
        
    model2 = models.Model(input=model1.input,output=x ) 
#    w_model2=list(model2.get_weights())
##    model2.set_weights(model1.get_weights()) 
    if (not(b_FC)):
        print("NotTrainable: [:-6]")

        for layer in model2.layers[:-6]:
            layer.trainable = False
    else:
        print("NotTrainable: [:-4]")
        for layer in model2.layers[:-4]:
            layer.trainable = False

#    print("Model2.Summary:")
#    model2.summary()
    model2.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
    
    #Extract features from the convolutional base  
    train_X = np.reshape(loadedImages_final, (shape0,shape1,shape2,shape3 ))
     
    model2.fit(train_X, label_final, batch_size=batch_size,epochs=epochs,verbose=1)
#    w_model2_fit=list(model2.get_weights())
    #model2.save('model2_complete.h5')
    model3 = models.Model(input=model2.input,output=model2.layers[-3].output)
    print("Model3.Summary:")
    model3.summary()

    print("\n")
    train_features = model3.predict(train_X)
    
    train_data_flatten = np.reshape(train_features, (train_features.shape[0], -1))
    return train_data_flatten
#%%
    
#%% Feature Extractor
#1- Most Entropy
def MostEntropyFeatures():
    
    #Asuume shape is (L1,L2,L3,L4) : (total_images,width,height,slices)
    #Each 3d image reshape to a row as (L1,L2*L3*L4), exp shape: (376,91*109*91)
#    print(train_data.shape)
    train_data_flatten=np.reshape(loadedImages_final,(loadedImages_final.shape[0], -1))
 #    print(train_data_flatten.shape)  
    return train_data_flatten
#%%
#2-Pretrained

#3- CNN _TrainFromScratch
def CNN():
    #Total params: 828,617
    shape0,shape1,shape2,shape3 = loadedImages_final.shape[0],loadedImages_final.shape[1],loadedImages_final.shape[2],loadedImages_final.shape[3]
    model = Sequential()
    model.add(Conv3D(8, (3, 3,3),padding='same', activation='relu',data_format='channels_last', input_shape= (shape1,shape2,shape3,1)))#(138,169,166,1) (176,220,205,1)
    model.add(Conv3D(8, (3, 3,3),padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2,2)))
    model.add(Dropout(25))
    model.add(Flatten())
    model.add(Dense(216, activation='relu'))
    model.add( Dense(1, activation='sigmoid')) 
    
    model.summary()
    #Extract features from the convolutional base  
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
    
    #Extract features from the convolutional base  
    train_features = np.reshape(loadedImages_final, (loadedImages_final.shape[0],loadedImages_final.shape[1],loadedImages_final.shape[2],loadedImages_final.shape[3],1))
    
    model.fit(train_features, label_final, batch_size=batch_size,epochs=epochs,verbose=1)

    train_features = model.predict(train_features)
    train_data_flatten = np.reshape(train_features, (train_features.shape[0], -1))
    return train_data_flatten



    

#%% Cross validation
#reference:https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

def Cross_validation(s_classifier,feature_flatten):
    s_classifier = 'svm-rbf'
    scores = {}
    X,y = feature_flatten,label_final
    
    #Classifier
    if s_classifier=='svm-rbf':
        classifier = SVC(kernel='rbf')
        #classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',  kernel='rbf', max_iter=1000, probability=False, random_state=None,  shrinking=True, tol=0.001, verbose=False) #SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',  kernel='rbf', max_iter=1000, probability=False, random_state=None,  shrinking=True, tol=0.001, verbose=False)
    elif s_classifier=='svm-linear':
        classifier =  LinearSVC(penalty='l2', loss='squared_hinge') #LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
    elif s_classifier =='rf':
        classifier = RandomForestClassifier()
    print('Classifier: ',classifier)
    
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    
    for train_index, test_index in cv.split(X,y):
#        print("Train Index: ", train_index, "\n")
#        print("Test Index: ", test_index)
    
        train_X,test_X,train_Y,test_Y  = X[train_index], X[test_index], y[train_index], y[test_index]
        numHC = len(train_Y [train_Y ==0])
        numMDD = len(train_Y [train_Y ==1])
        print ("Train numHC,numMDD: ", numHC,",", numMDD)
        numHC = len(test_Y [test_Y ==0])
        numMDD = len(test_Y [test_Y ==1])
        print ("Test numHC,numMDD: ", numHC,",", numMDD)
        
        train_X = Normalization (train_X)
#        train_data,train_ground = Augmentation(train_X,train_Y,'duplicate')
        train_X,train_Y = Augmentation(train_X,train_Y ,'duplicate')
        
#        print('No Augmentation')
        test_X = Normalization (test_X)
        
        
       
        #Classifier
        t0=time()
        train_X = train_X.reshape((len(train_X),feature_flatten.shape[1]))
        test_X = test_X.reshape((len(test_X),feature_flatten.shape[1]))    

        classifier.fit(train_X, train_Y)  
        trainClassifier_time=round(time()-t0, 3)
#        print ("training time:", round(time()-t0, 3), "s") # the time would be round to 3 decimal in seconds
        t1=time()     
        preds = classifier.predict(test_X)
        test_time=round(time()-t1, 3)
#        print ("predict time:", round(time()-t1, 3), "s")
        
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
        scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y, classifier.predict(train_X)))
        scores.setdefault('train_time', []).append(trainClassifier_time)
        scores.setdefault('test_time', []).append(test_time)
        scores.setdefault('test_tp', []).append(tps)
        scores.setdefault('test_tn', []).append(tns)
        scores.setdefault('test_fp', []).append(fps)
        scores.setdefault('test_fn', []).append(fns)

    for key, value in scores.items():
        scores[key] = np.array(value)
    return scores




#%% 
t2=time()
#s_features = 'MostEntropy'
#s_features = 'CNN_scratch'
s_features = 'Fine_tuned'
#s_features = 'Fine_tuned_inception'

#Feature Extractor
if s_features == 'MostEntropy':
    feature_flatten = MostEntropyFeatures()
elif s_features == 'CNN_scratch':
    feature_flatten = CNN()
elif s_features == 'Fine_tuned':
    feature_flatten = FineTuned_Pretrained3(0)
elif s_features == 'Fine_tuned_inception':
    feature_flatten = FineTuned_Pretrained3_Inception(0)
t3=time() 
featureExtractiontime=t3-t2   
print('Features: ',s_features)

#%%
scores1 = Cross_validation('svm-rbf',feature_flatten)
print ("Scores1:" , scores1)

scores2 = Cross_validation('svm-linear',feature_flatten)
print ("Scores2:" , scores2)

#print(scores1)


 #%%Evaluation
sens_ = np.round([
        scores1['test_sensitivity'].mean(),
        scores2['test_sensitivity'].mean()]
#       scores3['test_sensitivity'].mean(),
#       scores4['test_sensitivity'].mean(),
#       scores5['test_sensitivity'].mean(),
#       scores6['test_sensitivity'].mean()]
#       scores7['test_sensitivity'].mean(),
#       scores8['test_sensitivity'].mean(),
#       scores9['test_sensitivity'].mean(),
#       scores10['test_sensitivity'].mean(),
#       scores11['test_sensitivity'].mean(),
#       scores12['test_sensitivity'].mean()]
,2)
    
spec_ = np.round([
        scores1['test_specificity'].mean(),
        scores2['test_specificity'].mean()]
#       scores3['test_specificity'].mean(),
#       scores4['test_specificity'].mean(),
#       scores5['test_specificity'].mean(),
#       scores6['test_specificity'].mean()]
#       scores7['test_specificity'].mean(),
#       scores8['test_specificity'].mean(),
#       scores9['test_specificity'].mean(),
#       scores10['test_specificity'].mean(),
#       scores11['test_specificity'].mean(),
#       scores12['test_specificity'].mean()]
    ,2)   
    
prec1_ = np.round([
        scores1['test_percision'].mean(),
       scores2['test_percision'].mean()]
#       scores3['test_percision'].mean(),
#       scores4['test_percision'].mean(),
#       scores5['test_percision'].mean(),
#       scores6['test_percision'].mean()]
#       scores7['test_percision'].mean(),
#       scores8['test_percision'].mean(),
#       scores9['test_percision'].mean(),
#       scores10['test_percision'].mean(),
#       scores11['test_percision'].mean(),
#       scores12['test_percision'].mean()]
    ,2)

areaUnderROC = np.round([
        scores1['test_AUC'].mean(),
       scores2['test_AUC'].mean()]
#       scores3['test_AUC'].mean(),
#       scores4['test_AUC'].mean(),
#       scores5['test_AUC'].mean(),
#       scores6['test_AUC'].mean()]
#       scores7['test_AUC'].mean(),
#       scores8['test_AUC'].mean(),
#       scores9['test_AUC'].mean(),
#       scores10['test_AUC'].mean(),
#       scores11['test_AUC'].mean(),
#       scores12['test_AUC'].mean()]
,2)

auccuracy_train = np.round([
        scores1['train_Accuracy'].mean(),
       scores2['train_Accuracy'].mean()]
#       scores3['train_Accuracy'].mean(),
#       scores4['train_Accuracy'].mean(),
#       scores5['train_Accuracy'].mean(),
#       scores6['train_Accuracy'].mean()]
#       scores7['train_Accuracy'].mean(),
#       scores8['train_Accuracy'].mean(),
#       scores9['train_Accuracy'].mean(),
#       scores10['train_Accuracy'].mean(),
#       scores11['train_Accuracy'].mean(),
#       scores12['train_Accuracy'].mean()]
    ,2)
    
auccuracy_test = np.round([
        scores1['test_Accuracy'].mean(),
       scores2['test_Accuracy'].mean()]
#       scores3['test_Accuracy'].mean(),
#       scores4['test_Accuracy'].mean(),
#       scores5['test_Accuracy'].mean(),
#       scores6['test_Accuracy'].mean()]
#       scores7['test_Accuracy'].mean(),
#       scores8['test_Accuracy'].mean(),
#       scores9['test_Accuracy'].mean(),
#       scores10['test_Accuracy'].mean(),
#       scores11['test_Accuracy'].mean(),
#       scores12['test_Accuracy'].mean()]
,2)
    
precision_test = np.round([
        scores1['test_prec'].mean(),
       scores2['test_prec'].mean()]
#       scores3['test_prec'].mean(),
#       scores4['test_prec'].mean(),
#       scores5['test_prec'].mean(),
#       scores6['test_prec'].mean()]
#       scores7['test_prec'].mean(),
#       scores8['test_prec'].mean(),
#       scores9['test_prec'].mean(),
#       scores10['test_prec'].mean(),
#       scores11['test_prec'].mean(),
#       scores12['test_prec'].mean()]
,2)
    
recall_test = np.round([
        scores1['test_rec'].mean(),
       scores2['test_rec'].mean()]
#       scores3['test_rec'].mean(),
#       scores4['test_rec'].mean(),
#       scores5['test_rec'].mean(),
#       scores6['test_rec'].mean()]
#       scores7['test_rec'].mean(),
#       scores8['test_rec'].mean(),
#       scores9['test_rec'].mean(),
#       scores10['test_rec'].mean(),
#       scores11['test_rec'].mean(),
#       scores12['test_rec'].mean()]
    ,2)
train_time = np.round([
        scores1['train_time'].mean()+ featureExtractiontime,
       scores2['train_time'].mean()+ featureExtractiontime]
#       scores3['train_time'].mean(),
#       scores4['train_time'].mean(),
#       scores5['train_time'].mean(),
#       scores6['train_time'].mean()]
#       scores7['train_time'].mean(),
#       scores8['train_time'].mean(),
#       scores9['train_time'].mean(),
#       scores10['train_time'].mean(),
#       scores11['train_time'].mean(),
#       scores12['train_time'].mean()]
    ,2)
    
test_time = np.round([
        scores1['test_time'].mean(),
       scores2['test_time'].mean()]
#       scores3['test_time'].mean(),
#       scores4['test_time'].mean(),
#       scores5['test_time'].mean(),
#       scores6['test_time'].mean()]
#       scores7['test_time'].mean(),
#       scores8['test_time'].mean(),
#       scores9['test_time'].mean(),
#       scores10['test_time'].mean(),
#       scores11['test_time'].mean(),
#       scores12['test_time'].mean()]
    ,2)

result=np.array([areaUnderROC,auccuracy_test,sens_,spec_,prec1_,precision_test,recall_test,test_time,train_time,auccuracy_train])
result=np.transpose(result)
print (result)

   