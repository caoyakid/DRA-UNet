# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:23:15 2021

@author: a0956
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from keras_unet.models.multiresunet import MultiResUnet
from keras_unet.models import custom_unet
from keras_unet.models.attention_unet_2d import Attention_U_Net_2D
from keras_unet.models.se_unet_2d import SE_U_Net_2D
from keras_unet.models.unet import U_Net_2D
from keras_unet.models.unet_plusplus import UNetPlusPlus
from keras_unet.models.resunet import ResUNet_2D
from keras_unet.models.fcn_vgg import FCN32_VGG16, FCN8_VGG16
from keras_unet.models.fc_densenet103 import FC_DenseNet103

from keras_unet.metrics import jacard, dice_coef
from keras_unet.losses import jaccard_distance
from keras_unet.utils import trainStep, evaluateModel, saveModel, plot_imgs

if __name__ == '__main__':
    
    ### Load Data ###
    mypath = "D:/datasets/trainV1"
    sid = os.listdir(mypath)
    
    x_data = []
    y_data = []
    for s in sid:
        img = cv2.imread('D:/datasets/trainV1/'+ s +'/images/'+ s + '.jpg', cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img,(256, 256), interpolation = cv2.INTER_CUBIC)
        x_data.append(resized_img)
    
        msk = cv2.imread('D:/datasets/trainV1/'+ s +'/masks/'+ s + '_gt.jpg', cv2.IMREAD_GRAYSCALE)
        resized_msk = cv2.resize(msk,(256, 256), interpolation = cv2.INTER_CUBIC)
        y_data.append(resized_msk)
        
    imgs_np = np.asarray(x_data)
    masks_np = np.asarray(y_data)
    
    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.asarray(masks_np, dtype=np.float32)/255 
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    
    ### Train/val split ###
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=3, shuffle = False) 
    
    y_train = np.round(y_train,0)
    y_val = np.round(y_val,0)
    
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)
    
    ### Implement your model ###
    # model = MultiResUnet(height=256, width=256, n_channels=3)
    
    input_shape = x_train[0].shape
    
    # model = custom_unet(    
    # input_shape,
    # filters=32,
    # use_batch_norm=True,
    # dropout=0.3,
    # dropout_change_per_layer=0.0,
    # num_layers=4)
    
    # model = Attention_U_Net_2D(image_shape = input_shape, activation='elu',
    #                     feature_maps=[16, 32, 64, 128, 256], depth=4,
    #                     drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, 
    #                     batch_norm=False, k_init='he_normal', loss_type="bce", 
    #                     optimizer="sgd", lr=0.002, n_classes=1)
    
    # 
    
    model = U_Net_2D(image_shape = input_shape , activation='elu', feature_maps=[16, 32, 64, 128, 256], 
              depth=4, drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, 
              batch_norm=False, k_init='he_normal', loss_type="bce", 
              optimizer="sgd", lr=0.002, n_classes=1)
    
    
    # model = ResUNet_2D(image_shape = input_shape, activation='elu', k_init='he_normal',
    #             drop_values=[0.1,0.1,0.1,0.1,0.1], batch_norm=False, 
    #             feature_maps=[16,32,64,128,256], depth=4, loss_type="bce", 
    #             optimizer="sgd", lr=0.001, n_classes=1)
    
    # model = SE_U_Net_2D(image_shape = input_shape, activation='elu', feature_maps=[16, 32, 64, 128, 256], 
    #             depth=4, drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, 
    #             batch_norm=False, k_init='he_normal', loss_type="bce", 
    #             optimizer="sgd", lr=0.002, n_classes=1)
    
    #model = FCN8_VGG16(image_shape = input_shape, n_classes=1, lr=0.1, optimizer="adam")
    
    # model = FC_DenseNet103(image_shape = input_shape, n_filters_first_conv=48, n_pool=4, 
    #                growth_rate=12, n_layers_per_block=5, dropout_p=0.2,
    #                loss_type="bce", optimizer="sgd", lr=0.001)
    
    model.summary()