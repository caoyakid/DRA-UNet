# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:07:13 2021

@author: a0956
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# from keras_unet.metrics import jacard, dice_coef, iou, sensitivity, precision, specificity
from keras_unet.utils import  plot_imgs, evaluate_result, pred_only_FOV, get_augmented
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# from tensorflow.keras.utils import plot_model



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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3, shuffle = False) 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=3, shuffle = False) 
    
    y_train = np.round(y_train,0)
    y_val = np.round(y_val,0)
    
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    
    train_gen = get_augmented(
    x_train, y_train, batch_size=5,
    data_gen_args = dict(
        # rescale = 1./255,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range = [0.5, 1.5],
        shear_range=0.5,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant' #reflect or wrap
    ))
    
    
    
    
    sample_batch = next(train_gen)
    xx, yy = sample_batch
    print(xx.shape, yy.shape)

    plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=5, figsize=6)
    
    
    ### Plot images + masks overlay ###
    # plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=5, figsize=6)
    # plot_imgs(org_imgs=x_val, mask_imgs=y_val, nm_img_to_plot=5, figsize=6)