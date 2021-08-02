# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:27:48 2021

@author: a0956
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras_unet.metrics import jacard, dice_coef, iou, sensitivity, precision, specificity
from keras_unet.utils import  plot_imgs, evaluate_result, pred_only_FOV
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=9, shuffle = False) 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=9, shuffle = False) 
    
    y_train = np.round(y_train,0)
    y_val = np.round(y_val,0)
    
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)
    
    ### Plot images + masks overlay ###
    # plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=5, figsize=6)
    # plot_imgs(org_imgs=x_val, mask_imgs=y_val, nm_img_to_plot=5, figsize=6)
    
    dependencies = {
    'jacard': jacard,
    'dice_coef': dice_coef,
    'iou': iou,
    'sensitivity': sensitivity,
    'precision': precision,
    'specificity': specificity}
    
    
    ### Loading the model ### method1
    # modelname = '0702models_UNet_v1_aug'
    # json_file = open(modelname + '/modelP.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    
    # model.load_weights(modelname + '/weightsW.h5')
    # test_border_masks = model.load_weights(modelname + '/weightsW.h5')
    
    ### Plot original + ground truth + pred + overlay (pred on top of original) ###
    # y_pred = loaded_model.predict(x_val)
    # y_pred = np.round(y_pred, 0)

    # plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=5)
    

    ### Loading the model ### method2
    modelname = '0716models_MultiResUnet_v1'
    model = load_model(modelname + '/weights.hdf5', custom_objects=dependencies) # hdf5 originally
    test_border_masks = model.load_weights(modelname + '/weights.hdf5')
    
    dirname = '0716results_MultiResUnet_v1'
    evaluate_result(model, x_test, y_test, modelname, dirname)
    
    ####### Test whether it can load multi-model and plot in one figure ####### 
    # if there is a # &&& mark: it means that extra step creating data to plot the ROC curve
    # &&&
    # modelname1 = '0620models_UNet_v1'
    # model1 = load_model(modelname1 + '/weights.hdf5', custom_objects=dependencies)
    # test_border_masks1 = model1.load_weights(modelname1 + '/weights.hdf5')
    
    
    y_pred = model.predict(x_test)
    # y_pred1 = model1.predict(x_test) # &&&
    
    ### Purpose: draw AUC ROC 
    y_test = np.round(y_test, 0) 
    # kill_border(y_pred, test_border_masks)
    y_scores, y_true = pred_only_FOV(y_pred, y_test , test_border_masks)#returns data only inside the FOV
    print("Calculating results only inside the FOV:")
    print("y scores pixels: " +str(y_scores.shape[0]) +"  including background around retina: " +str(y_pred.shape[0]*y_pred.shape[2]*y_pred.shape[3]))
    print("y true pixels: " +str(y_true.shape[0]) +"  including background around retina: " +str(y_test.shape[2]*y_test.shape[3]*y_test.shape[0]))
    
    # &&&
    # y_scores1, y_true1 = pred_only_FOV(y_pred1, y_test , test_border_masks1)
    # print("Calculating results only inside the FOV:")
    # print("y scores pixels: " +str(y_scores1.shape[0]) +"  including background around retina: " +str(y_pred1.shape[0]*y_pred1.shape[2]*y_pred1.shape[3]))
    # print("y true pixels: " +str(y_true1.shape[0]) +"  including background around retina: " +str(y_test.shape[2]*y_test.shape[3]*y_test.shape[0]))
    
    
    #Area under the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    
    # fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_scores1) # &&&
    # AUC_ROC1 = roc_auc_score(y_true1, y_scores1) # &&&
    
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    # print("\nArea under the ROC curve: " +str(AUC_ROC1)) # &&&
    roc_curve =plt.figure()
    plt.plot(fpr,tpr,'-',label='ROC (AUC = %0.4f)' % AUC_ROC)
    # plt.plot(fpr1,tpr1,'-r',label='ROC (AUC = %0.4f)' % AUC_ROC1) # &&&
    plt.title('ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True,linestyle = "-",color = 'gray' ,linewidth = '0.2',axis='both')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    # plt.savefig("ROC.png")
    plt.savefig(modelname + '/ROC.png',format='png')
    plt.close()
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision,recall)
    print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    # plt.savefig("Precision_recall.png")
    plt.savefig(modelname + '/Precision_recall.png',format='png')
    plt.close()
    
    y_pred = np.round(y_pred, 0) # 取四捨五入
       
    plot_imgs(org_imgs=x_test, mask_imgs=y_test, pred_imgs=y_pred, nm_img_to_plot=10)
    
    # plot_model(model, "plot_model_UNet.png", show_shapes=True)
    
    
    
    
