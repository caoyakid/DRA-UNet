# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:06:50 2021

@author: a0956
"""

"""
version information
python == 3.7
Nvidia 2070s
CUDA 
cudnn
tensorflow-gpu 2.0.0
tensorboard 2.2.1
numpy 1.19.2
cv2 4.2.0.32

"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from keras_unet.models.multiresunet import MultiResUnet
from keras_unet.models import custom_unet
from keras_unet.models.attention_unet_2d import Attention_U_Net_2D
from keras_unet.models.unet import U_Net_2D
from keras_unet.models.unet_plusplus import UNetPlusPlus
from keras_unet.models.resunet import ResUNet_2D
from keras_unet.models.fcn_vgg import FCN32_VGG16, FCN8_VGG16
from keras_unet.models.fc_densenet103 import FC_DenseNet103
from keras_unet.models.dra_unet import DRA_UNet


from keras_unet.metrics import jacard, dice_coef, iou, sensitivity, precision, specificity
# from keras_unet.losses import jaccard_distance
from keras_unet.utils import plot_segm_history_iou, plot_segm_history_acc, plot_segm_history_dice , get_augmented
from keras_unet.utils import plot_segm_history_SE , plot_segm_history_PC, plot_segm_history_SP
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=9, shuffle = True) 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=9, shuffle = True) 
    
    y_train = np.round(y_train,0)
    y_val = np.round(y_val,0)
    
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    
    # Data augmentation
    # train_gen, val_gen = get_augmented(
    # x_train, y_train, x_val, y_val, batch_size=2,
    # data_gen_args = dict(
    #      rescale = 1./255,
    #     # featurewise_center=True,
    #     # featurewise_std_normalization=True,
    #     # samplewise_center=True,
    #     # samplewise_std_normalization=True,
    #     rotation_range=0.,
    #     width_shift_range=0.,
    #     height_shift_range=0.,
    #     brightness_range = [0.5, 1.5],
    #     shear_range=0.,
    #     zoom_range=0.1,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     fill_mode='constant' #reflect or wrap
    # ))
    
    ### Implement your model ###
    
    input_shape = x_train[0].shape
    
    
    # model = U_Net_2D(image_shape = input_shape , activation='elu', feature_maps=[16, 32, 64, 128, 256], 
    #           depth=4, drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=False, 
    #           batch_norm=True, k_init='he_normal', loss_type="bce", 
    #           optimizer="sgd", lr=0.002, n_classes=1)
    
    # model = custom_unet(input_shape,filters=32,use_batch_norm=True,dropout=0.3,dropout_change_per_layer=0.0,num_layers=4)
    
    # model = FCN8_VGG16(image_shape = input_shape, n_classes=1, lr=0.05, optimizer="adam") #lr=0.1
    
    # model = FC_DenseNet103(image_shape = input_shape, n_filters_first_conv=32, n_pool=4, 
    #                 growth_rate=16, n_layers_per_block=5, dropout_p=0.2,
    #                 loss_type="bce", optimizer="sgd", lr=0.05) #lr = 0.001 growth_rate = 12
    
    # model = ResUNet_2D(image_shape = input_shape, activation='elu', k_init='he_normal',
    #         drop_values=[0.1,0.1,0.1,0.1,0.1], batch_norm=True, 
    #         feature_maps=[16,32,64,128,256], depth=4, loss_type="bce", 
    #         optimizer="sgd", lr=0.01, n_classes=1)
    
    # model = Attention_U_Net_2D(image_shape = input_shape, activation='elu',
                        # feature_maps=[16, 32, 64, 128, 256], depth=4,
                        # drop_values=[0.1,0.1,0.2,0.2,0.3], spatial_dropout=True, 
                        # batch_norm=True, k_init='he_normal', loss_type="bce", 
                        # optimizer="sgd", lr=0.002, n_classes=1)

    
    # model = MultiResUnet(height=256, width=256, n_channels=3)

    model = DRA_UNet(height=256, width=256, n_channels=3)
    

  
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou, dice_coef, 'acc', 
                                                                         sensitivity, precision, specificity])
    
    modelname = '0717models_DRA_UNet_v1'
    os.makedirs(modelname)
    # saveModel(model, modelname)
  
    # dirname = 'TT0617results_UNet'
    # os.makedirs(dirname)
    # trainStep(model, x_train, y_train, x_val, y_val, epochs=50, batchSize=2, modelname=modelname, dirname=dirname)
    batch_size = 4
    steps_per_epoch = np.ceil(len(x_train)/batch_size)
    checkpointer = ModelCheckpoint(filepath='D:/DL/mycode/' + modelname + '/weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=100, steps_per_epoch=steps_per_epoch, verbose=1, validation_data=(x_val, y_val), callbacks=[checkpointer])
    
    # history = model.fit_generator(
    #     train_gen,
    #     steps_per_epoch = 300,
    #     epochs = 100,
    #     verbose=1, 
    #     callbacks=[checkpointer],
    #     validation_data=val_gen,
    #     validation_steps=100,
    #     )
    
    
    
    plot_segm_history_acc(history, modelname, metrics=["acc", "val_acc"])
    plot_segm_history_iou(history,modelname, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss']) 
    plot_segm_history_dice(history,modelname, metrics=["dice_coef", "val_dice_coef"])
    plot_segm_history_SE(history,modelname, metrics=["sensitivity", "val_sensitivity"])
    plot_segm_history_PC(history,modelname, metrics=["precision", "val_precision"])
    plot_segm_history_SP(history,modelname, metrics=["specificity", "val_specificity"])
    
    # model_json = model.to_json()
    # fp = open(modelname + '/modelP.json','w')
    # fp.write(model_json)
    # model.save_weights(modelname + '/weightsW.h5')
    



    

    