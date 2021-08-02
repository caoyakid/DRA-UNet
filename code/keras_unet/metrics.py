# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:26:41 2021

@author: a0956
"""

import tensorflow as tf

from keras_unet import TF
if TF:
    from tensorflow.keras import backend as K
else:    
    from keras import backend as K
    
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def acc(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC


def sensitivity(y_true, y_pred): # = recall
    """ recall """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP/(TP + FN + K.epsilon())
    return SE


def precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP/(TP + FP + K.epsilon())
    return PC


def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP



# # define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
# def auc_roc(y_true, y_pred):
#     # any tensorflow metric
#     value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

#     # find all variables created for this metric
#     metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

#     # Add metric variables to GLOBAL_VARIABLES collection.
#     # They will be initialized for new session.
#     for v in metric_vars:
#         tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

#     # force to update metric values
#     with tf.control_dependencies([update_op]):
#         value = tf.identity(value)
#         return value

# class RocCallback(Callback):
#     def __init__(self,training_data,validation_data):
#         self.x = training_data[0]
#         self.y = training_data[1]
#         self.x_val = validation_data[0]
#         self.y_val = validation_data[1]


#     def on_train_begin(self, logs={}):
#         return

#     def on_train_end(self, logs={}):
#         return

#     def on_epoch_begin(self, epoch, logs={}):
#         return

#     def on_epoch_end(self, epoch, logs={}):
#         y_pred_train = self.model.predict_on_batch(self.x)
#         roc_train = roc_auc_score(self.y, y_pred_train)
#         y_pred_val = self.model.predict_on_batch(self.x_val)
#         roc_val = roc_auc_score(self.y_val, y_pred_val)
#         print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
#         return

#     def on_batch_begin(self, batch, logs={}):
#         return

#     def on_batch_end(self, batch, logs={}):
#         return