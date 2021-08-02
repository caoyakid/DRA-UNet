# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 03:01:40 2021

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

def DenseBlock(channels,inputs):


    conv1_1 = Conv2D(channels, (1, 1),activation=None, padding='same')(inputs)
    conv1_1=BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same')(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv3) #activation = None
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv3_1) #activation = None
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv4) #activation = None
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv4_1) #activation = None
    conv4_2 = BatchActivate(conv4_2)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('elu')(x) #relu
    return x

def DRA_UNet(height, width, n_channels):
    inputs = Input((height, width, n_channels))

    desblock1 = DenseBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(desblock1)
    desblock1 = ResPath(32, 4, desblock1)

    desblock2 = DenseBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(desblock2)
    desblock2 = ResPath(32*2, 3, desblock2)

    desblock3 = DenseBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(desblock3)
    desblock3 = ResPath(32*4, 2, desblock3)

    desblock4 = DenseBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(desblock4)
    desblock4 = ResPath(32*8, 1, desblock4)

    desblock5 = DenseBlock(32*16, pool4)
    
    v_desblock5 = Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(desblock5)
    attn1 = AttentionBlock(v_desblock5, desblock4, 16, True)
    result1 = concatenate([attn1, v_desblock5])
    desblock6 = DenseBlock(32*8, result1)
    
    v_desblock6 = Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same')(desblock6)
    attn2 = AttentionBlock(v_desblock6, desblock3, 32, True)
    result2 = concatenate([attn2, v_desblock6])
    desblock7 = DenseBlock(32*4, result2)
    
    v_desblock7 = Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same')(desblock7)
    attn3 = AttentionBlock(v_desblock7, desblock2, 64, True)
    result3 = concatenate([attn3, v_desblock7])
    desblock8 = DenseBlock(32*2, result3)
    
    v_desblock8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(desblock8)
    attn4 = AttentionBlock(v_desblock8, desblock1, 128, True)
    result4 = concatenate([attn4, v_desblock8])
    desblock9 = DenseBlock(32, result4)
    
    conv10 = conv2d_bn(desblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model

# def DenseUNet(input_size=(512, 512, 3), start_neurons=16, keep_prob=0.9,block_size=7,lr=1e-3):

#     inputs = Input(input_size)
#     conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
#     conv1 = BatchActivate(conv1)
#     conv1 = DenseBlock(start_neurons * 1, conv1)
#     pool1 = MaxPooling2D((2, 2))(conv1)

#     conv2 = DenseBlock(start_neurons * 2, pool1)
#     pool2 = MaxPooling2D((2, 2))(conv2)

#     conv3 = DenseBlock(start_neurons * 4, pool2)
#     pool3 = MaxPooling2D((2, 2))(conv3)


#     convm = DenseBlock(start_neurons * 8, pool3)


#     deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
#     uconv3 = concatenate([deconv3, conv3])
#     uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same")(uconv3)
#     uconv3 = BatchActivate(uconv3)
#     uconv3 = DenseBlock(start_neurons * 4, uconv3)


#     deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
#     uconv2 = concatenate([deconv2, conv2])
#     uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same")(uconv2)
#     uconv2 = BatchActivate(uconv2)
#     uconv2 = DenseBlock(start_neurons * 2, uconv2)

#     deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#     uconv1 = concatenate([deconv1, conv1])
#     uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same")(uconv1)
#     uconv1 = BatchActivate(uconv1)
#     uconv1 = DenseBlock(start_neurons * 1, uconv1)

#     output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
#     output_layer = Activation('sigmoid')(output_layer_noActi)

#     model = Model(input=inputs, output=output_layer)

#     model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

#     return model