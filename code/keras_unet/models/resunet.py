# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:37:48 2021

@author: a0956
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dropout, Lambda, SpatialDropout2D, Conv2D, \
                                    Conv2DTranspose, MaxPooling2D, Concatenate,\
                                    Add, BatchNormalization, ELU
# from metrics import binary_crossentropy_weighted, jaccard_index, \
#                     weighted_bce_dice_loss


def ResUNet_2D(image_shape, activation='elu', k_init='he_normal',
               drop_values=[0.1,0.1,0.1,0.1,0.1], batch_norm=False, 
               feature_maps=[16,32,64,128,256], depth=4, loss_type="bce", 
               optimizer="sgd", lr=0.001, n_classes=1):

    """Create 2D Residual_U-Net.
       Parameters
       ----------
       image_shape : array of 3 int
           Dimensions of the input image.
       activation : str, optional
           Keras available activation type.
       k_init : str, optional
           Keras available kernel initializer type.
       drop_values : array of floats, optional
           Dropout value to be fixed.
       batch_norm : bool, optional
           Use batch normalization.
       feature_maps : array of ints, optional
           Feature maps to use on each level. Must have the same length as the 
           ``depth+1``.
       
       depth : int, optional
           Depth of the network.                        
                                                                           
       loss_type : str, optional
           Loss type to use, three type available: ``bce`` (Binary Cross Entropy),
           ``w_bce`` (Weighted BCE, based on weigth maps) and ``w_bce_dice``
           (Weighted loss: ``weight1*BCE + weight2*Dice``). 
                                                                           
       optimizer : str, optional
           Optimizer used to minimize the loss function. Posible options: 
           ``sgd`` or ``adam``.                         
                                                                           
       lr : float, optional
           Learning rate value.
       n_classes: int, optional                                                 
           Number of classes.  
       Returns
       -------
       model : Keras model
           Model containing the U-Net.
       Calling this function with its default parameters returns the following
       network:
       .. image:: img/resunet.png
           :width: 100%
           :align: center
       Where each green layer represents a residual block as the following:
        
       .. image:: img/res_block.png
           :width: 45%
           :align: center
       Images created with `PlotNeuralNet <https://github.com/HarisIqbal88/PlotNeuralNet>`_.
    """

    if len(feature_maps) != depth+1:                                            
        raise ValueError("feature_maps dimension must be equal depth+1")
    if len(drop_values) != depth+1:
        raise ValueError("'drop_values' dimension must be equal depth+1")

    fm = feature_maps[::-1]

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    inputs = Input(dinamic_dim)

    x = level_block(inputs, depth, fm, 3, activation, k_init, drop_values, 
                    batch_norm, True)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid') (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    # # Select the optimizer
    # if optimizer == "sgd":
    #     opt = tf.keras.optimizers.SGD(
    #         lr=lr, momentum=0.99, decay=0.0, nesterov=False)
    # elif optimizer == "adam":
    #     opt = tf.keras.optimizers.Adam(
    #         lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
    #         amsgrad=False)
    # else:
    #     raise ValueError("Error: optimizer value must be 'sgd' or 'adam'")

    # # Compile the model
    # if loss_type == "bce":
    #     model.compile(optimizer=opt, loss='binary_crossentropy',
    #                   metrics=[jaccard_index])
    # elif loss_type == "w_bce":
    #     model.compile(optimizer=opt, loss=binary_crossentropy_weighted(weights),
    #                   metrics=[jaccard_index])
    # elif loss_type == "w_bce_dice":
    #     model.compile(optimizer=opt,
    #                   loss=weighted_bce_dice_loss(w_dice=0.66, w_bce=0.33),
    #                   metrics=[jaccard_index])
    # else:
    #     raise ValueError("'loss_type' must be 'bce', 'w_bce' or 'w_bce_dice'")

    return model


def level_block(x, depth, f_maps, f_size, activation, k_init, drop_values,   
                batch_norm, first_block):                                       

    """Produces a level of the network. It calls itself recursively.
                                                                                
       Parameters
       ----------
       x : Keras layer
           Input layer of the block.                          
                                                                           
       depth :int
           Depth of the network. This value determines how many times the
           function will be called recursively.
       f_maps : array of ints
           Feature maps to use.                        
                                                                           
       f_size : int
           Convolution window.                                                             
                                                                           
       activation : str, optional
           Keras available activation type.        
                                                                           
       k_init : str, optional
           Keras available kernel initializer type.    
                                                                           
       drop_values : array of floats, optional
           Dropout value to be fixed.
                                                                           
       batch_norm : bool, optional
           Use batch normalization.                       
                                                                           
       first_block : float, optional
           To advice the function that it is the first residual block of the
           network, which avoids Full Pre-Activation layers.                                              
                                                                                
       Returns
       -------                                                                 
       x : Keras layer
           last layer of the levels.
    """  
                                                                                
    if depth > 0:                                                               
        r = residual_block(x, f_maps[depth], f_size, activation, k_init,           
                           drop_values[depth], batch_norm, first_block)                 
        x = MaxPooling2D((2, 2)) (r)                                         
        x = level_block(x, depth-1, f_maps, f_size, activation, k_init, 
                        drop_values, batch_norm, False)                          
        x = Conv2DTranspose(f_maps[depth], (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])                                               
        x = residual_block(x, f_maps[depth], f_size, activation, k_init,           
                           drop_values[depth], batch_norm, False)                       
    else:                                                                       
        x = residual_block(x, f_maps[depth], f_size, activation, k_init,           
                           drop_values[depth], batch_norm, False)                       
    return x 


def residual_block(x, f_maps, f_size, activation='elu', k_init='he_normal', 
                   drop_value=0.0, bn=False, first_block=False):                                          

    """Residual block.
                                                                                
       Parameters
       ----------
       x : Keras layer
           Input layer of the block.
                                                                                
       f_maps : array of ints
           Feature maps to use.
            
       f_size : int
           Convolution window. 
       activation : str, optional
           Keras available activation type.        
                                                                                
       k_init : str, optional
           Keras available kernel initializer type.    
                                                                                
       drop_value : float, optional
           Dropout value to be fixed. 
                                                                                
       bn : bool, optional
           Use batch normalization.               
                                                                                
       first_block : float, optional
           To advice the function that it is the first residual block of the
           network, which avoids Full Pre-Activation layers.
                                                                                
       Returns
       -------
       x : Keras layer
           Last layer of the block.
    """ 
                                                                                
    # Create shorcut                                                            
    shortcut = Conv2D(f_maps, activation=None, kernel_size=(1, 1),           
                      kernel_initializer=k_init)(x)                             
                                                                                
    # Main path                                                                 
    if not first_block:                                                         
        x = BatchNormalization()(x) if bn else x                                
        if activation == 'elu':
            x = ELU(alpha=1.0) (x)                                                  
                                                                                
    x = Conv2D(f_maps, f_size, activation=None,                            
               kernel_initializer=k_init, padding='same') (x)                   
                                                                                
    x = Dropout(drop_value) (x) if drop_value > 0 else x                        
    x = BatchNormalization()(x) if bn else x                                    
    if activation == 'elu':
        x = ELU(alpha=1.0) (x)                                                      
                                                                                
    x = Conv2D(f_maps, f_size, activation=None,                            
               kernel_initializer=k_init, padding='same') (x)                   
    x = BatchNormalization()(x) if bn else x                                    
                                                                                
    # Add shortcut value to main path                                           
    x = Add()([shortcut, x])                                                    
                                                                                
    return x         