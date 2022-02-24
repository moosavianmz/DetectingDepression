# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:51:25 2020

@author: mmousa4
"""



#%%import 
import os
import pandas as pd
import tensorflow as tf
from keras.backend import tensorflow_backend as K, resize_volumes
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


#%% Loading data
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#maps_ICA_resized_indv= np.load('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections\\maps_ICA_resized_indv_323232.npy')
os. chdir('C:\\Users\\mmousa4\\OneDrive - Louisiana State University\\Final_codes\\Sections')

#final_labels = np.load('final_labels.npy')
#img_resized_indv= np.load('ts_aal.npy')
#img_resized_indv= np.load('ts_smith.npy')



#maps_ICA_resized_indv= np.load('maps_ICA_resized_indv_323232.npy')
#final_labels = np.load('final_labels_323232_balanced.npy')
#np.save('final_labels_balanced.npy',final_labels)
#indx_rus = np.load('indx_rus_9_pc.npy')
#final_labels_balanced=final_labels[indx_rus]

#img_resized_indv= np.load('signal_3d_resized_indv_323232_balanced.npy')
#img_resized_indv_balanced=img_resized_indv[indx_rus]#.reshape(-1,img_resized_indv.shape[1]*img_resized_indv.shape[2])
#np.save('signal_3d_resized_indv_646431_balanced.npy',img_resized_indv)
#canicaGL_covariences= np.load('ts_canicaGL_covariences.npy')
#canicaGL_covariences = canicaGL_covariences[indx_rus]

img_resized_indv_balanced = np.load('signal_adhd_3d_resized_indv_323232.npy')
final_labels_balanced = np.load('adhd_labels.npy')


#%% 3D Feature Extractor
from keras import layers,models,optimizers,regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def Vgg16_Seq (train_X,train_Y,test_X,test_Y):
    #Qureshi, Muhammad Naveed Iqbal, Jooyoung Oh, and Boreom Lee. "3D-CNN based discrimination of schizophrenia using resting-state fMRI." Artificial intelligence in medicine 98 (2019): 10-17
    img_count,img_rows, img_cols, img_depth=train_X.shape[0],train_X.shape[1],train_X.shape[2],train_X.shape[3]
    num_classes=n_components
    img_input= layers.Input(shape=(img_rows, img_cols, img_depth, 1))
   
    
       # Block 1
    x = layers.Conv3D(64, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv3D(64, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv3D(128, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(256, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(512, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2,2), strides=(2, 2,2), name='block5_pool')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
  
    # Create model.
    model = models.Model(img_input, x, name='vgg16')

    tr = np.reshape(train_X,train_X.shape + (1,))
    te = np.reshape(test_X,test_X.shape + (1,))
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)   
    print (model.summary())
    model.compile(optimizer=optimizers.Adam(lr=00.1), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(tr, train_Y, batch_size=16,epochs=5,verbose=1,validation_data=(te, test_Y),callbacks=[es])
    
    train_features = model.predict(tr)
    test_features= model.predict(te)
    
    
    #Apply PCA
    #--------------
#    from sklearn.decomposition import PCA

#    pca = PCA()
#    train_features_pca = pca.fit_transform(train_features)
#    test_features_pca = pca.transform(test_features)
#      train_features , test_features  = train_features_pca, test_features_pca

#    print("PCA Applied")
    
    train_data_flatten = np.reshape(train_features, (train_features.shape[0], -1))
    test_data_flatten = np.reshape(test_features, (test_features.shape[0], -1))
    
#    train_data_flatten = np.reshape(train_features, (train_features.shape[0], -1))
#    test_data_flatten = np.reshape(test_features, (test_features.shape[0], -1))
    
    np.save('train_data_flatten.npy', train_data_flatten)
    np.save('test_data_flatten.npy', test_data_flatten)
#    d = np.load('test3.npy')
    return train_data_flatten, test_data_flatten



def Zhao_CNN3D(Xtrain,Ytrain,Xtest,Ytest):
#    Reference: Zhao, Yu, et al. "Automatic recognition of fMRI-derived functional networks using 3-D convolutional neural networks." IEEE Transactions on Biomedical Engineering 65.9 (2017): 1975-1984.
    
#    train_3d,final_labels_train_3d[:,0],test_3d,final_labels_test_3d[:,0]
    IMAGE_HEIGHT = Xtrain.shape[1]
    IMAGE_WIDTH = Xtrain.shape[2]
    IMAGE_DEPTH = Xtrain.shape[3]
    img_shape = layers.Input(shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1 ))
#    img_shape =(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1 )


     
    x = layers.Conv3D(32, (5,5,5),
                      activation='relu',
                      strides=(2, 2, 2),
                      padding='same',
                      name='block1_conv1')(img_shape)
    x = (layers.LeakyReLU(alpha=.01))(x)
    x = layers.Conv3D(32, (3, 3,3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = (layers.LeakyReLU(alpha=.01))(x)
    x = layers.MaxPooling3D((2, 2,2), name='block1_pool')(x)
    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    model = models.Model(img_shape, x, name='CNN3D')
    model.compile(optimizer = optimizers.Adam(lr=00.1), loss='binary_crossentropy', metrics=['accuracy'])

    
    model.summary()
    
    
    # Fit data to model
    train_v = Xtrain.reshape(-1,Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3],1)
    test_v = Xtest.reshape(-1,Xtest.shape[1],Xtest.shape[2],Xtest.shape[3],1)

    es = EarlyStopping(monitor='loss', mode='min', verbose=1) 
    history = model.fit(train_v,Ytrain,
                batch_size=16,
                epochs=5,
                verbose=1,
                validation_split=0.3)
      
    train_preds = model.predict(train_v)
    test_preds= model.predict(test_v)   
    np.save ('Zhao_CNN3D_train_preds.npy',train_preds)
    np.save ('Zhao_CNN3D_test_preds.npy',test_preds)
    
    print ("#Zhao_CNN3D_train output shape", str(train_preds.shape)) 
    print ("#Zhao_CNN3D_test output shape", str(test_preds.shape)) 
           
    train_preds= np.squeeze (np.round(train_preds).astype('int'))
    test_preds= np.squeeze (np.round(test_preds).astype('int') )
       
    return train_preds, test_preds

def C3DNN(Xtrain,Ytrain,Xtest,Ytest):
    #https://towardsdatascience.com/step-by-step-implementation-3d-convolutional-neural-network-in-keras-12efbdd7b130
    
#    Xtrain,Ytrain,Xtest,Ytest = train_3d,final_labels_train_3d[:,0],test_3d,final_labels_test_3d[:,0] 
    IMAGE_HEIGHT = Xtrain.shape[1]
    IMAGE_WIDTH = Xtrain.shape[2]
    IMAGE_DEPTH = Xtrain.shape[3]
#    img_shape = layers.Input(shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1 ))
    img_shape =(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1 )
   
    # Create the model
    model = models.Sequential()
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=img_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization(center=True, scale=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization(center=True, scale=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    
   
    
    # Fit data to model
    train_v = Xtrain.reshape(-1,Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3],1)
    test_v = Xtest.reshape(-1,Xtest.shape[1],Xtest.shape[2],Xtest.shape[3],1)

    history = model.fit(train_v,Ytrain,
                batch_size=32,
                epochs=5,
                verbose=1,
                validation_split=0.3)
    
    
    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('3D lstm loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#    plt.savefig('LSTM3D_lossplot.png')    


    train_preds = model.predict(train_v)
    test_preds= model.predict(test_v)   
    
    np.save ('C3DNN_train_preds.npy',train_preds)
    np.save ('C3DNN_test_preds.npy',test_preds)
    
    print ("#C3DNN_train output shape", str(train_preds.shape)) 
    print ("#C3DNN_test output shape", str(test_preds.shape)) 
    
#    ax1.scatter(test_preds[Ytest==1],)
    train_preds= np.squeeze (np.round(train_preds).astype('int'))
    test_preds= np.squeeze (np.round(test_preds).astype('int') )
       
    return train_preds, test_preds
    

def Li_C3D (train_X,train_Y,test_X,test_Y):#Convolution 3D CNN
    #Li, Xiaoxiao, et al. "2-channel convolutional 3D deep neural network (2CC3D) for fMRI analysis: ASD classification and feature learning." 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018). IEEE, 2018.
    IMAGE_HEIGHT = train_X.shape[1]
    IMAGE_WIDTH = train_X.shape[2]
    IMAGE_DEPTH = train_X.shape[3]
    img_shape = layers.Input(shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1 ))
    
    conv1 = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding= 'same', activation='relu')(inputs= img_shape)	
    maxpool1 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding= 'same')(conv1)
    dropout1 = layers.Dropout (0.5)(maxpool1)
    conv2 = layers.Conv3D(filters=64, kernel_size=(3,3,3), padding= 'same', activation='relu')(dropout1)	
    maxpool2 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding= 'same')(conv2)
    dropout2 = layers.Dropout (0.5)(maxpool2)
    conv3a = layers.Conv3D(filters=128, kernel_size=(3,3,3), padding= 'same', activation='relu')(dropout2)
#    conv3b = layers.Conv3D(filters=128, kernel_size=(3,3,3), padding= 'same', activation='relu')(conv3a)
    maxpool3 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding= 'same')(conv3a)
    dropout3 = layers.Dropout (0.5)(maxpool3)
    conv4a = layers.Conv3D(filters=128, kernel_size=(3,3,3), padding= 'same', activation='relu')(dropout3)
#    conv4b = layers.Conv3D(filters=128, kernel_size=(3,3,3), padding= 'same', activation='relu')(conv4a)
    maxpool4 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding= 'same')(conv4a)
    dropout4 = layers.Dropout(0.5)(maxpool4)
    flatten = layers.Flatten()(dropout4)
    dense1 = layers.Dense( units=256,activation='relu',kernel_regularizer=regularizers.l2 (0.01))(flatten)
    dense2 = layers.Dense( units=256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense3 = layers.Dense( units=1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(dense2)

    model = models.Model(input= img_shape, output= dense3) 
    model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
#    utils.print_summary(model, line_length=100, positions=[.30, .65, .77, 1.], print_fn=None)
    model.summary()

    es = EarlyStopping(monitor='loss', mode='min', verbose=1) 
    train_v = train_X.reshape(-1,train_X.shape[1],train_X.shape[2],train_X.shape[3],1)
    test_v = test_X.reshape(-1,test_X.shape[1],test_X.shape[2],test_X.shape[3],1)

    history = model.fit(train_v,train_Y, epochs=10, batch_size=32)#validation_split=0.2,, shuffle=True,callbacks=[es])
    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('3D lstm loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#    plt.savefig('LSTM3D_lossplot.png')    


    train_preds = model.predict(train_v)
    test_preds= model.predict(test_v)
    train_preds= np.squeeze (np.round(train_preds).astype('int'))
    test_preds= np.squeeze (np.round(test_preds).astype('int') )

        
    print ("#TrainFeatures shape", str(train_preds.shape)) 
    print ("#TestFeatures shape", str(test_preds.shape))   
          
   
    return train_preds, test_preds



def srinjaypaul_CAE_3D(volumes_train,volumes_test):#Convolutional Auto encoder
    #Reference:https://srinjaypaul.github.io/3D_Convolutional_autoencoder_for_brain_volumes/
    volumes_train,volumes_test = train_3d,test_3d
    mini_batch = 32
    batch_size = volumes_train.shape[0]
    IMAGE_HEIGHT = volumes_train.shape[1]
    IMAGE_WIDTH = volumes_train.shape[2]
    IMAGE_DEPTH = volumes_train.shape[3]
    
    inputs_ =  IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1 
    img_shape = layers.Input(shape = (inputs_))
    
     #encoder
    conv1 = layers.Conv3D(filters=16, kernel_size=(3,3,3), padding= 'same', activation='relu')(inputs= img_shape)	
    maxpool1 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding= 'same')(conv1)
    conv2 = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding= 'same', activation='relu')(maxpool1)	
    maxpool2 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding= 'same')(conv2)
    conv3 = layers.Conv3D(filters=96, kernel_size=(2,2,2), padding= 'same', activation='relu')(maxpool2)	
    maxpool3 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding= 'same')(conv3)
	  #latent internal representation

	  #decoder
    unpool1 = layers.UpSampling3D((2,2,2))(maxpool3) 
    deconv1 = layers.Conv3D( filters=96, kernel_size=(2,2,2), padding= "same" ,  activation='relu')(unpool1)
    unpool2 = layers.UpSampling3D((2,2,2))(deconv1) 
    deconv2 = layers.Conv3D( filters=32, kernel_size=(3,3,3), padding= "same" ,  activation='relu')(unpool2)
    unpool3 = layers.UpSampling3D((2,2,2))(deconv2) 
    deconv3 = layers.Conv3D( filters=16, kernel_size=(3,3,3), padding= "same" ,  activation='relu')(unpool3)
    decoder = layers.Dense( units=1,activation='sigmoid')(deconv3)

#    print("shape of decoded {}".format(K.int_shape(decoder)))
     
    autoencoder = models.Model(input= img_shape, output= decoder) 
    encoder = models.Model(img_shape, maxpool3)

    autoencoder.compile(optimizer = optimizers.RMSprop(),loss='mean_squared_error')
    
    encoder.summary()    
    autoencoder.summary()    
    

    es = EarlyStopping(monitor='loss', mode='min', verbose=1)  
#    train_data = train_data.reshape(-1, 28,28, 1)
   
    volumes_train = volumes_train.reshape (-1,volumes_train.shape[1],volumes_train.shape[2],volumes_train.shape[3],1)
    volumes_test = volumes_test.reshape (-1,volumes_test.shape[1],volumes_test.shape[2],volumes_test.shape[3],1)
    history = autoencoder.fit(volumes_train,volumes_train, epochs=5, batch_size=mini_batch, shuffle=True,callbacks=[es], validation_split=0.2)
   
    autoencoder = autoencoder.save_weights('autoencoder_mri.h5')
    encoder = encoder.save_weights('encoder_mri.h5')
    
    # loading weights of a keras model
#    autoencoder.load_weights('autoencoder_mri.h5')
#    encoder.load_weights('encoder_mri.h5')
#    # loading whole model
#    from keras.models import load_model
#    autoencoder = load_model('autoencoder_mri.h5')

    # summarize history for loss
    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#    plt.savefig('CAE3D_lossplot.png')    
    
    encoded_img_train = encoder.predict(volumes_train)
    encoded_img_test = encoder.predict(volumes_test)
#    decoder_img = decoder.predict(encoded_img_train)
  #    decoder = decoder.save_weights('decoder_mri.h5')
    
#    test_features_ts= encoder.predict(test_X_)
    print ("Train Features CAE:",encoded_img_train.shape)
    print ("Test Features CAE:",encoded_img_test.shape)
#    train_features_ts = np.squeeze(train_features_ts,axis=4)
#    test_features_ts = np.squeeze(test_features_ts,axis=4)
    
    np.save('srinjaypaul_CAE_3D_trainpreds.npy',encoded_img_train)
    np.save('srinjaypaul_CAE_3D_testpreds.npy',encoded_img_test)
    return encoded_img_train,encoded_img_test

#%%
def CAE_1D_1(train_ts,train_gt, test_ts,test_gt):
    #Denoising Convolutional Autoencoder (DCAE)
    #Compare with traditional DAE DCAE the same basic structure of encoder and decoder but replaces the fully-connected layers with convolutional 
#    train_ts,train_gt, test_ts,test_gt = timeseries_1D_train,pool_timeseries_aal[train_index], timeseries_1D_test,pool_timeseries_aal[test_index]
    # ENCODER
    input_sig = layers.Input(batch_shape=(None,120,1))
    x = layers.Conv1D(8,3, activation='relu', padding='same')(input_sig)
    x1 = layers.MaxPooling1D(2)(x)
    x2 = layers.Conv1D(16,5, activation='relu', padding='same')(x1)
    x3 = layers.MaxPooling1D(2)(x2)
   
    x4 = layers.Conv1D(32,1, activation='relu', padding='same')(x3)
    encoder = models.Model(input_sig, x4)
     
    # DECODER 
    x3_ = layers.Conv1D(16, 5, activation='relu', padding='same')(x4)
    x2_ = layers.UpSampling1D(2)(x3_)
    x1_ = layers.Conv1D(8, 3, activation='relu', padding='same')(x2_)
    x_ = layers.UpSampling1D(2)(x1_)
    decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x_)
   
    print("shape of decoded {}".format(K.int_shape(decoded)))
     
    autoencoder = models.Model(input_sig, decoded)
#    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    autoencoder.compile(optimizer = optimizers.RMSprop(),loss='mean_squared_error')
    autoencoder.summary()
#    train_data = train_data.reshape(-1, 28,28, 1)
    train_X_ = train_ts[:,:,np.newaxis]
    test_X_ = test_ts[:,:,np.newaxis]
    es = EarlyStopping(monitor='loss', mode='min', verbose=1)  
    autoencoder.fit(train_X_,train_X_, epochs=100, batch_size=16, shuffle=True,callbacks=[es])
    train_features_ts = encoder.predict(train_X_)
    test_features_ts= encoder.predict(test_X_)
    print ("Train Features ts:",train_features_ts.shape)
    print ("Test Features ts:",test_features_ts.shape)
#    train_features_ts = np.squeeze(train_features_ts,axis=4)
#    test_features_ts = np.squeeze(test_features_ts,axis=4)
    return train_features_ts,test_features_ts


#%%Unet 2D, https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = layers.MaxPooling2D((2, 2)) (c1)
    p1 = layers.Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = layers.MaxPooling2D((2, 2)) (c2)
    p2 = layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = layers.MaxPooling2D((2, 2)) (c3)
    p3 = layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = layers.Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = layers.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    u9 = layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = models.Model(inputs=[input_img], outputs=[outputs])
    return model

def UNET_2D(trainX,trainY,testX,testY):
#    trainX,trainY,testX,testY = train_3d,np.mean(train_3d,axis=3),test_3d,np.mean(test_3d,axis=3)
    IMAGE_HEIGHT = trainX.shape[1]
    IMAGE_WIDTH = trainX.shape[2]
    IMAGE_DEPTH = trainX.shape[3]
    
    inputs_ =  IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH 
    input_img = layers.Input(inputs_, name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    
    model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
    trainY=trainY.reshape((-1,32,32,1))
    results = model.fit(trainX,trainY, batch_size=32, epochs=10, callbacks=callbacks, validation_split=0.2)
    # Predict on train, val and test
    preds_train = model.predict(trainX, verbose=1)
    preds_test = model.predict(testX, verbose=1)
    
    # Threshold predictions
#    preds_train_t = (preds_train > 0.5).astype(np.uint8)
#    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    print ("Train Features CAE:",preds_train.shape)
    print ("Test Features CAE:",preds_test.shape)
    
    preds_train = np.squeeze(preds_train)
    preds_test = np.squeeze(preds_test)
    return preds_train,preds_test


    
#%%Pretrained VGG16 2D
def grayToRBG (loadedImages_gray):
    Covert2RGB='yes'
    if Covert2RGB=='yes':
        loadedImages_RGB=np.zeros((loadedImages_gray.shape[0],loadedImages_gray.shape[1],loadedImages_gray.shape[2],3)) #Initilize entropy
       
        for i in range (0,loadedImages_gray.shape[0]):
            img_RGB=gray2rgb(loadedImages_gray[i,:,:])
            loadedImages_RGB[i,:,:,:]=img_RGB.copy()
    
    else:
        loadedImages_RGB=loadedImages_gray.copy()
       
    
    return loadedImages_RGB


from keras.applications import VGG16,inception_v3,DenseNet121,Xception
from skimage.color import gray2rgb
def FineTuned_Pretrained(train_X,train_Y,test_X, test_Y):
    b_FC =1    
    shape0,height, width, channels=train_X.shape[0],train_X.shape[1],train_X.shape[2],3
        
    print("FineTuned_Pretrained_VGG")
    base_model  = VGG16(weights='imagenet',include_top=False, input_shape=(height, width, channels))
    x = layers.Flatten(name='flatten')(base_model.output)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
#    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(input=base_model.input,output=predictions ) 
    
    
    
#    print ("DenseNet")
#    base_model  = DenseNet121(weights='imagenet',include_top=False, input_shape=(height, width, channels))
#    x = base_model.layers[-1].output  # 
#    x = layers.GlobalAveragePooling2D()(x)
#    prepredictions = layers.Dense(256, activation='relu')(x)
#    predictions = layers.Dense(1, activation='sigmoid')(prepredictions)
#    model = models.Model(input=base_model.input,output=predictions ) 
    
    
#    print ("Xception")
#    base_model = Xception(weights='imagenet', include_top=False,input_shape=(height, width, channels))
#    x = base_model.output
#    x = layers.GlobalAveragePooling2D()(x)
#    prepredictions = layers.Dense(256, activation='relu')(x)
#    predictions = layers.Dense(1, activation='sigmoid')(prepredictions)
#    model = models.Model(input=base_model.input,output=predictions )

    
   
    if (not(b_FC)):
        print("NotTrainable: [:10]")

        for layer in model.layers[:10]:
            layer.trainable = False
    else:
        print("NotTrainable: [:-4]")
        for layer in model.layers[:-4]:
            layer.trainable = False

    print("Model.Summary:")
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
    
    #fit the model. Extract features from the convolutional base 
    train_X_g =  grayToRBG(train_X)
    test_X_g =  grayToRBG(test_X)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1)  
    model.fit(train_X_g, train_Y, batch_size=16,epochs=10,verbose=1)#,callbacks=[es])
    
    #Classification Part
    #--------------------
    preds_test = model.predict(test_X_g)
    preds_train=model.predict(train_X_g)
    # Threshold predictions
    train_preds_t= np.squeeze (np.round(preds_train).astype('int'))
    test_preds_t= np.squeeze (np.round(preds_test).astype('int') )
    
#    
    return train_preds_t, test_preds_t
#    
    
    #Feature Extractor Part
    #-----------------------
#    model_FE = models.Model(input=model.input,output=model.layers[-2].output)
#    print("VGG16_FeatureExtractor.Summary:")
#    model_FE.summary()
#    
#    train_features = model_FE.predict(train_X_g )
#    test_features= model_FE.predict(test_X_g )        
#    
#    return train_features, test_features


#%%
def Expand_2D(train_X,train_Y,test_X,test_Y):
     #Train Data
    #--------------------
    train_2d=np.zeros((train_X.shape[0]*train_X.shape[3]*train_X.shape[4],train_X.shape[1],train_X.shape[2]),dtype='float16') #Initilize entropy
    final_labels_train_2d=np.zeros((train_X.shape[0]*train_X.shape[3]*train_X.shape[4],2),dtype='uint8') #Initilize entropy
    
    n=0 #Counter
    for i in range (0,train_X.shape[0]):
        for j in range (0,train_X.shape[4]):
            for k in range (0,train_X.shape[3]):
                train_2d[n,:,:]=train_X[i,:,:,k,j].copy()
                final_labels_train_2d[n]=train_Y[i].copy(),i
                n=n+1
    
     #Test Data
    #--------------------
    test_2d=np.zeros((test_X.shape[0]*test_X.shape[3]*test_X.shape[4],test_X.shape[1],test_X.shape[2]),dtype='float16') #Initilize entropy
    final_labels_test_2d=np.zeros((test_X.shape[0]*test_X.shape[3]*test_X.shape[4],2),dtype='uint8') #Initilize entropy
    
    n=0 #Counter
    for i in range (0,test_X.shape[0]):
        for j in range (0,test_X.shape[4]):
            for k in range (0,test_X.shape[3]):
                test_2d[n,:,:]=test_X[i,:,:,k,j].copy()
                final_labels_test_2d[n]=test_Y[i].copy(),i
                n=n+1
    return train_2d,final_labels_train_2d,test_2d,final_labels_test_2d
    
def ExpandData(train_X,train_Y,test_X,test_Y):
    
    #Train Data
    #--------------------
    train_3d=np.zeros((train_X.shape[0]*train_X.shape[4],train_X.shape[1],train_X.shape[2],train_X.shape[3]),dtype='float16') #Initilize entropy
    final_labels_train_3d=np.zeros((train_X.shape[0]*train_X.shape[4],2),dtype='uint8') #Initilize entropy
    
    n=0 #Counter
    for i in range (0,train_X.shape[0]):
        for j in range (0,train_X.shape[4]):
            train_3d[n,:,:,:]=train_X[i,:,:,:,j].copy()
            final_labels_train_3d[n]=train_Y[i].copy(),i
            n=n+1

    
    #Test data
    #-------------------
    test_3d=np.zeros((test_X.shape[0]*test_X.shape[4],test_X.shape[1],test_X.shape[2],test_X.shape[3]),dtype='float16') #Initilize entropy
    final_labels_test_3d=np.zeros((test_X.shape[0]*test_X.shape[4],2),dtype='uint8') #Initilize entropy
    
    n=0 #Counter
    for i in range (0,test_X.shape[0]):
        for j in range (0,test_X.shape[4]):
            test_3d[n,:,:,:]=test_X[i,:,:,:,j].copy()
            final_labels_test_3d[n]=test_Y[i].copy(),i
            n=n+1
    
    return train_3d,final_labels_train_3d,test_3d,final_labels_test_3d
#%%Balancing the dataset
#x,y=img_resized_indv,final_labels
#x1=img_resized_indv[final_labels==1]
#x0=img_resized_indv[final_labels==0]
#img_resized_indv_balanced=np.concatenate((x1,x0),axis=0)
#final_labels_balanced=np.concatenate(([1]*x1.shape[0],[0]*x0.shape[0]),axis=0)

#%% Confusion Matrix
from sklearn.metrics import roc_curve, auc,accuracy_score, confusion_matrix, f1_score, roc_auc_score,precision_score,recall_score
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0,1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1,0]

#%% Cross Validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR,SVC
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut,StratifiedKFold,train_test_split,LeaveOneOut
from imblearn import under_sampling ,over_sampling
import statistics

#CV method
#-------------
cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)#GroupShuffleSplit(n_splits=3)#GroupKFold()#LeaveOneGroupOut()
#cv = LeaveOneOut()
print ("CrossValidation:", cv)


x,y=img_resized_indv_balanced,final_labels_balanced

scores,scores1 = {},{}
for train_index, test_index in cv.split(x,y):#,groups=final_labels[:,1]):
    train_X,test_X,train_Y,test_Y  = x[train_index], x[test_index], y[train_index], y[test_index]
#    train_2d,test_2d = train_X,test_X
    train_3d,final_labels_train_3d,test_3d,final_labels_test_3d= ExpandData(train_X,train_Y,test_X,test_Y)
#    train_2d,final_labels_train_3d,test_2d,final_labels_test_3d= Expand_2D(train_X,train_Y,test_X,test_Y)
    
    #Feature Extractor
    ######################### 
#    train_2d,test_2d = UNET_2D(train_3d,np.mean(train_3d,axis=3),test_3d,np.mean(test_3d,axis=3))
#    train_feature,test_feature = FineTuned_Pretrained(train_X,train_Y,test_X, test_Y)
#    train_feature,test_feature = FineTuned_Pretrained(train_2d,final_labels_train_3d[:,0],test_2d, final_labels_test_3d[:,0])
    
    
#    encoded_img_train,encoded_img_test = srinjaypaul_CAE_3D(train_3d,test_3d)
#    train_2d,test_2d = UNET_2D(encoded_img_train,final_labels_train_3d,encoded_img_test,final_labels_test_3d)


    #Augmentation
    ########################
#    from collections import Counter
#    print("Train labels:",Counter(train_Y))
  
#    print("Train labels:",Counter(final_labels_train_3d))
#    oversample = ADASYN()
#        oversample = SMOTE()
#        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
#        rus = over_sampling.RandomOverSampler(random_state=42)
#
#    rus = under_sampling.RandomUnderSampler(random_state=42)
#    train_X, train_Y = rus.fit_sample(train_feature, final_labels_train_3d[:,0])
          
    
    
    #Classifier
    ########################
#    train_preds,preds = Li_C3D (train_feature,final_labels_train_3d[:,0],test_feature,final_labels_test_3d[:,0])

#    train_preds,preds = Li_C3D (train_3d,final_labels_train_3d[:,0],test_3d,final_labels_test_3d[:,0])
#    train_preds,preds = C3DNN (train_3d,final_labels_train_3d[:,0],test_3d,final_labels_test_3d[:,0])
    train_preds,preds = Zhao_CNN3D (train_3d,final_labels_train_3d[:,0],test_3d,final_labels_test_3d[:,0])#encoded_img_train,final_labels_train_3d[:,0],encoded_img_test,final_labels_test_3d[:,0])
#    train_preds,preds = FineTuned_Pretrained(train_2d,train_Y,test_2d, test_Y)
#    train_preds,preds = FineTuned_Pretrained(train_2d,final_labels_train_3d[:,0],test_2d, final_labels_test_3d[:,0])
#   train_preds,test_preds = C3DNN_train_preds,C3DNN_test_preds

#    train_X,test_X=encoded_img_train.reshape ((encoded_img_train.shape[0],-1)),encoded_img_test.reshape((encoded_img_test.shape[0],-1))
#    
#    classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
##    classifier.fit(train_feature,final_labels_train_3d[:,0])  
#    classifier.fit(train_feature,train_Y) 
#    train_preds = classifier.predict(train_feature)
#    preds = classifier.predict(test_feature)
    
    
    
#    preds_train = (preds_train >= 0.5).astype(np.uint8)
#    preds = (preds >= 0.5).astype(np.uint8)
    
#    preds_train= np.squeeze (np.round(train_preds).astype('int'))
#    preds= np.squeeze (np.round(test_preds).astype('int') )
    
    #Performance Evaluation
    #######################    
    true_labels= final_labels_test_3d[:,0]
#    true_labels = test_Y
    scores.setdefault('test_AUC', []).append(roc_auc_score(true_labels,preds))
    scores.setdefault('test_Accuracy', []).append(accuracy_score(true_labels, preds))
    tps=tp(true_labels, preds)
    tns = tn(true_labels, preds)
    fps=fp(true_labels, preds)
    fns= fn(true_labels, preds)
    sensitivity= tps/(tps+fns)
    specificity=tns/(tns+fps)
    percision=tps/(tps+fps)
    scores.setdefault('test_sensitivity', []).append(sensitivity)
    scores.setdefault('test_specificity', []).append(specificity)
    scores.setdefault('test_percision', []).append(percision)
#    scores.setdefault('test_prec', []).append(precision_score(true_labels, preds))
    scores.setdefault('test_rec', []).append(recall_score(true_labels, preds))
#    scores.setdefault('train_Accuracy', []).append(accuracy_score(train_Y, train_preds))
    scores.setdefault('train_Accuracy', []).append(accuracy_score(final_labels_train_3d[:,0], train_preds))
    scores.setdefault('test_tp', []).append(tps)
    scores.setdefault('test_tn', []).append(tns)
    scores.setdefault('test_fp', []).append(fps)
    scores.setdefault('test_fn', []).append(fns)
    print(scores)

    #Second level classifier (Majority vote)
    ########################
    from collections import Counter
    
    groups = (final_labels_test_3d[:,1]).astype(int)
    print(final_labels_test_3d)
    mydict = list(zip(preds, groups))
    
    df = pd.DataFrame(mydict, columns =['Pred', 'Group']) 
    print(df)
    pred2 = df.groupby('Group')['Pred'].apply(lambda x: x.value_counts().index[0]).reset_index()
    print(pred2)
    test_Y,preds = test_Y,pred2['Pred'].values
    print("Label:",test_Y)
    print("---------")
    print("Pred:",preds)
    
    #Final Results
    #--------------
    print("#Majority Vote")
    scores1.setdefault('test_AUC', []).append(roc_auc_score(test_Y,preds))
    scores1.setdefault('test_Accuracy', []).append(accuracy_score(test_Y, preds))
    tps=tp(test_Y, preds)
    tns = tn(test_Y, preds)
    fps=fp(test_Y, preds)
    fns= fn(test_Y, preds)
    sensitivity= tps/(tps+fns)
    specificity=tns/(tns+fps)
    percision=tps/(tps+fps)
    scores1.setdefault('test_sensitivity', []).append(sensitivity)
    scores1.setdefault('test_specificity', []).append(specificity)
    scores1.setdefault('test_percision', []).append(percision)
    scores1.setdefault('test_prec', []).append(precision_score(test_Y, preds))
    scores1.setdefault('test_rec', []).append(recall_score(test_Y, preds))
#    scores1.setdefault('train_Accuracy', []).append(accuracy_score(train_Y_a, classifier.predict(train_data_flatten)))
#    scores1.setdefault('train_Accuracy', []).append(accuracy_score(train_Y, classifier.predict(train_data_flatten)))
    scores1.setdefault('test_tp', []).append(tps)
    scores1.setdefault('test_tn', []).append(tns)
    scores1.setdefault('test_fp', []).append(fps)
    scores1.setdefault('test_fn', []).append(fns)
    

print(scores)    
print ("*Mean Performance: sensitivity,specificity,AUC,Accuracy,precision,recall")
print([statistics.mean(scores['test_sensitivity']),statistics.mean(scores['test_specificity']),statistics.mean(scores['test_AUC']),statistics.mean(scores['test_Accuracy']),statistics.mean(scores['test_percision']),statistics.mean(scores['test_rec'])])
print ("###########################################################")
