# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 04:22:20 2021

@author: a0956
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,\
                                    concatenate, BatchNormalization, Activation, \
                                    add, Add, Multiply
from tensorflow.keras.models import Model

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x

def MultiResBlock(U, inp, alpha = 1.67):
    
    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out

def ResPath(filters, length, inp):
    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out

def AttentionBlock(x, shortcut, filters, batch_norm):
    g1 = Conv2D(filters, kernel_size = 1)(shortcut) 
    g1 = BatchNormalization() (g1) if batch_norm else g1
    x1 = Conv2D(filters, kernel_size = 1)(x) 
    x1 = BatchNormalization() (x1) if batch_norm else x1

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1, kernel_size = 1)(psi) 
    psi = BatchNormalization() (psi) if batch_norm else psi
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x,psi])
    return x

def mra_unet(height, width, n_channels):
    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)
    
    v_mresblock5 = Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5)
    attn1 = AttentionBlock(v_mresblock5, mresblock4, 16, True)
    mresblock6 = MultiResBlock(32*8, attn1)
    
    v_mresblock6 = Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6)
    attn2 = AttentionBlock(v_mresblock6, mresblock3, 32, True)
    mresblock7 = MultiResBlock(32*4, attn2)
    
    v_mresblock7 = Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7)
    attn3 = AttentionBlock(v_mresblock7, mresblock2, 64, True)
    mresblock8 = MultiResBlock(32*2, attn3)
    
    v_mresblock8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8)
    attn4 = AttentionBlock(v_mresblock8, mresblock1, 128, True)
    mresblock9 = MultiResBlock(32, attn4)
    
    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model