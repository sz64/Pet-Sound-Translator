from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import librosa as lb
import pandas as pd

from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU#, ReLU
from keras.layers.core import Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.models import Model

sr=16000
N_FFT      = 512
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 30

#The pretrained network defined in paper: https://arxiv.org/abs/1703.09179
def CNN(weights='msd', input_tensor=None,include_top=True):
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1876)
        #input_shape = (1, 96, 1366)  #51550 for 180 sec
    else:
        input_shape = (96, 1876, 1)
        #input_shape = (96, 1366, 1)  #10 second audio length and sr (16k=626 and 12k = 469)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_1')(x)
    #x = ELU()(x)
    #x = ReLU()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_1')(x)

    # Conv block 2
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_2')(x)
    #x = ELU()(x)
    #x = ReLU()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_2')(x)

    # Conv block 3
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_3')(x)
    #x = ELU()(x)
    #x = ReLU()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_3')(x)
    
    # Conv block 4
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_4')(x)
    #x = ELU()(x) 
    #x = ReLU()(x)  #Check both bottomneck feature and Fine tuning
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_4')(x)
    
    # Conv block 5
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_5')(x)
    
    if include_top:
        x = GlobalMaxPooling2D(name='MP')(x)
        x = Dense(50, activation='sigmoid')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        weights_path = './model_best.hdf5'
        model.load_weights(weights_path, by_name=True)

        return model

#Data preprocessing
#---------------------------------------------------------------------------------------------------------------------#
def log_scale_melspectrogram(path):
    signal, sr_n = lb.load(path, sr=sr)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*sr_n)
    
    # cliping or padding
    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*sr_n) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
        
    melspect = lb.core.amplitude_to_db(lb.feature.melspectrogram(y=signal, sr=sr_n, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2)
    
    return melspect 

def get_melspectrograms(labels_dense=None):
    if labels_dense == None:
        labels_file  = './CatSound_Dataset.csv' 
        labels = pd.read_csv(labels_file,header=0)
        labels_dense = labels
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path']])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms

#-------------------------------------------------------------------------------------------#
def main():
    print('loading the model and the pre-trained weights...')
    #The softmax layer is similar so include_top=True
    base_model = CNN(weights='msd', include_top=False)

    #labels = rd.get_labels()
    X_train = get_melspectrograms()
    print("The input train mel-spectrogram size:", X_train.shape)

    save_feature_path = './CNN_Features/'
    features = base_model.predict(X_train)
    print("Saving model features")
    #Feature save on .npy format
    np.save(open(save_feature_path + 'CNN_Top_FC_features.npy', 'wb'), features)
    #-------------------------------------------------------------------------------------------#
    '''
    print("Extractiong all layer Concatenated GAP Features")
    #Individial or all layer feature extraction after GAM
    layer1_GAP = GAP2D()(base_model.get_layer('conv2d_1').output)
    layer2_GAP = GAP2D()(base_model.get_layer('conv2d_2').output)
    layer3_GAP = GAP2D()(base_model.get_layer('conv2d_3').output)
    layer4_GAP = GAP2D()(base_model.get_layer('conv2d_4').output)
    layer5_GAP = GAP2D()(base_model.get_layer('conv2d_5').output)
    feat_all = concat([layer1_GAP, layer2_GAP,layer3_GAP, layer4_GAP, layer5_GAP]) 

    GAP_ALL = Model(inputs=base_model.input, outputs=feat_all)
    Out_GAP_All = GAP_ALL.predict(X_train)

    np.save(open(save_feature_path +'CNN_Conv_5Layer_GAP_features.npy', 'wb'), Out_GAP_All)

    '''
    #-------------------------------------------------------------------------------------------#
    #Individial layer feature extraction
    feat_layer1 = base_model.get_layer('conv2d_1').output
    print('First Layer output shape:',K.int_shape(feat_layer1) )

    print("Extractiong Layer_1 feature")
    feat_extractor1 = Model(input=base_model.input, output=feat_layer1)
    Output_layer1 = feat_extractor1.predict(X_train)

    np.save(open(save_feature_path +'Layer_1_features.npy', 'wb'), Output_layer1)
    #-------------------------------------------------------------------------------------------#
    feat_layer2 = base_model.get_layer('conv2d_2').output
    print('Second Layer output shape:',K.int_shape(feat_layer2) )

    print("Extractiong Layer_2 feature")
    feat_extractor2 = Model(input=base_model.input, output=feat_layer2)
    Output_layer2 = feat_extractor2.predict(X_train)

    np.save(open(save_feature_path +'Layer_2_features.npy', 'wb'), Output_layer2)
    #-------------------------------------------------------------------------------------------#

    feat_layer3 = base_model.get_layer('conv2d_3').output
    print('Third Layer output shape:',K.int_shape(feat_layer3) )

    print("Extractiong Layer_3 feature")
    feat_extractor3 = Model(input=base_model.input, output=feat_layer3)
    Output_layer3 = feat_extractor3.predict(X_train)

    np.save(open(save_feature_path +'Layer_3_features.npy', 'wb'), Output_layer3)
    #-------------------------------------------------------------------------------------------#
    feat_layer4 = base_model.get_layer('conv2d_4').output
    print('Fourth Layer output shape:',K.int_shape(feat_layer4) )

    print("Extractiong Layer_4 feature")
    feat_extractor4 = Model(input=base_model.input, output=feat_layer4)
    Output_layer4 = feat_extractor4.predict(X_train)

    np.save(open(save_feature_path +'Layer_4_features.npy', 'wb'), Output_layer4)

    #-------------------------------------------------------------------------------------------#
    feat_layer5 = base_model.get_layer('conv2d_5').output
    print('Fifth Layer output shape:',K.int_shape(feat_layer5) )

    print("Extractiong Layer_5 feature")
    feat_extractor5 = Model(input=base_model.input, output=feat_layer5)
    Output_layer5= feat_extractor5.predict(X_train)

    np.save(open(save_feature_path +'Layer_5_features.npy', 'wb'), Output_layer5)
    #-------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()