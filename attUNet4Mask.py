# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:55:31 2021

@author: user
"""
from Data4Mask import saveResult, testGenerator, dataGenerator
import numpy as np 
import skimage.filters as filters
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, EarlyStopping
from PIL import Image



gpus= tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)


def batch_rgb2bianry(dir_str):
    im_prefix = ['.jpg','.tiff','.png']
    images = os.listdir(dir_str)
    # images.sort(key = lambda x:int(x.split('_')[0]))
    img_No = 0
    for filename in images:
        prefix = os.path.splitext(filename)[1]
        img_name = os.path.splitext(filename)[0]
        if prefix in im_prefix:
            img = Image.open(dir_str + filename)
            img = img.convert('L')
            threshold = filters.threshold_otsu(np.array(img))
            table = []
            for i in range(256):
                if i <= threshold:
                    table.append(0)
                else:
                    table.append(1)
            im_binary = img.point(table, '1')
            im_binary.save(dir_str+img_name+'.png')

" ================================ Unet ================================"
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    
    #----------------part 1-----------------#
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization(momentum=0.99)(conv1)
    conv1 = LeakyReLU(alpha=0.05)(conv1)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization(momentum=0.99)(conv1)
    conv1 = LeakyReLU(alpha=0.05)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization(momentum=0.99)(conv2)
    conv2 = LeakyReLU(alpha=0.05)(conv2)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization(momentum=0.99)(conv2)
    conv2 = LeakyReLU(alpha=0.05)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization(momentum=0.99)(conv3)
    conv3 = LeakyReLU(alpha=0.05)(conv3)
    conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization(momentum=0.99)(conv3)
    conv3 = LeakyReLU(alpha=0.05)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization(momentum=0.99)(conv4)
    conv4 = LeakyReLU(alpha=0.05)(conv4)
    conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization(momentum=0.99)(conv4)
    conv4 = LeakyReLU(alpha=0.05)(conv4)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
   
    conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization(momentum=0.99)(conv5)
    conv5 = LeakyReLU(alpha=0.05)(conv5) 
    conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization(momentum=0.99)(conv5)
    conv5 = LeakyReLU(alpha=0.05)(conv5) 
    
    #----------------part 2-----------------#
    up6 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.05)(conv6) 
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.05)(conv6) 

    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.05)(conv7)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.05)(conv7) 

    up8 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.05)(conv8) 
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.05)(conv8) 
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.05)(conv9) 
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.05)(conv9) 
    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9) # maybe can change the initializer to glorot_normal(xavier)
    conv9 = LeakyReLU(alpha=0.05)(conv9) 
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9) # maybe can change it to Tanh

    model = Model(inputs, conv10)
    

    model.compile(optimizer = Adam(lr = 5e-4), loss = binary_crossentropy, metrics = ['accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
" ========================================================================"


" =============================attention Unet============================="
def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate




# Attention U-Net
def att_unet(pretrained_weights = None,input_shape = (256,256,1), data_format='channels_last'):
    inputs = Input(input_shape)
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = LeakyReLU(alpha=0.05)(x)
        #x = ELU(alpha=1)(x) # 1009 new
        x = Conv2D(features, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = LeakyReLU(alpha=0.05)(x)
        #x = ELU(alpha=1)(x) # 1009 new
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        features = features * 2

    x = Conv2D(features, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = LeakyReLU(alpha=0.05)(x) 
    #x = ELU(alpha=1)(x) # 1009 new
    x = Conv2D(features, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = LeakyReLU(alpha=0.05)(x) 
    #x = ELU(alpha=1)(x) # 1009 new

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization(momentum=0.99)(x) # 1009 new
        x = LeakyReLU(alpha=0.05)(x)
        #x = ELU(alpha=1)(x) # 1009 new
        x = Conv2D(features, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization(momentum=0.99)(x) # 1009 new
        x = LeakyReLU(alpha=0.05)(x) 
        #x = ELU(alpha=1)(x) # 1009 new

    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    conv9 = BatchNormalization(momentum=0.99)(conv9) # 1009 new
    conv9 = LeakyReLU(alpha=0.05)(conv9) 
    #conv9 = ELU(alpha=1)(conv9) # 1009 new
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr = 5e-4), loss = BCE_SSIM_Loss, metrics = ['accuracy', 'mse'])
    
    # load pretrained_weights if needed
    if(pretrained_weights):
    	  model.load_weights(pretrained_weights)
    return model
"======================================================================"


class callback_alpha(Callback):
    def __init__(self, alp):
        self.alp = alp
    def on_epoch_end(self, epoch, logs=None):
        if (epoch > 10): 
            K.print_tensor(alp)
            K.set_value(self.alp, np.clip(self.alp-0.05, 0.7, 1))    

alp = K.variable(0.9) 
def BCE_SSIM_Mixed(y_true, y_pred):
    bce_loss = K.mean(K.binary_crossentropy(y_true, y_pred))
    ssim_loss = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=7, filter_sigma=1.5, k1=0.01, k2=0.03)
    return alp*bce_loss + (1.0-alp)*(1.0 - K.mean(ssim_loss))


def BCE_SSIM_Loss(y_true, y_pred):
    loss = tf.py_function(func = BCE_SSIM_Mixed,
                             inp=[y_true, y_pred],
                             Tout=tf.float32)
    return loss


if __name__ == '__main__':
        
    
    # #-----------------------------------training(bp formart)----------------------------------#
    # # image augment
    data_gen_args = dict(rescale=1./255,
                    rotation_range=5,
                    shear_range=0.02,
                    zoom_range=0.2,
                    width_shift_range=0.02,
                    height_shift_range=0.02,
                    horizontal_flip=True,
                    vertical_flip=True,
                    validation_split=0.2,
                    fill_mode='reflect')
    bs = 8 # batch size
    trainGene, valGene, tn, vn = dataGenerator(bs,'E:\\Code_v1\\forD1\\','cell_image','cell_mask',data_gen_args,save_to_dir = None) # bs/training set folder/image folder/mask folder/augment params/augmented image save folder
    model = att_unet() # weights of last round 
    # # save the best model weight
    model_checkpoint_w = ModelCheckpoint('E:\\Code_v2\\model\\weight4test\\', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, factor=0.5, min_lr=1e-5) #降低学习率
    earlystop = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit_generator(trainGene, steps_per_epoch=tn*5, epochs=1, validation_data=valGene, validation_steps=tn*5, callbacks=[model_checkpoint_w, earlystop, learning_rate_reduction, callback_alpha(alp)]) 
    # model.summary()

    ## save history accuracy and loss
    accy=history.history['accuracy']
    lossy = history.history['loss']
    val_accy=history.history['val_accuracy']
    val_lossy = history.history['val_loss']
    np_accy = np.array(accy).reshape((1,len(accy))) 
    np_lossy =np.array(lossy).reshape((1,len(lossy)))
    np_valaccy = np.array(val_accy).reshape((1,len(val_accy))) 
    np_vallossy =np.array(val_lossy).reshape((1,len(val_lossy)))
    plt.figure(1,dpi=300)
    plt.plot(accy,label='accuracy')
    plt.plot(val_accy,label='val_accuracy')
    plt.legend()
    plt.figure(2,dpi=300)
    plt.plot(lossy,label='lossy')
    plt.plot(val_lossy,label='val_lossy')
    plt.legend()
    np_out = np.concatenate([np_accy,np_lossy,np_valaccy,np_vallossy],axis=0)
    np.savetxt('E:\\Code_v2\\model_hist.txt',np_out.T)    

    # ----------------------------------------------------#
    
    # # ---------------------------------- test ---(own data)-------------------------------#
    # # own data
    # testGene = testGenerator('E:\\Code_v1\\training_pool\\')
    # model = att_unet()
    # model.load_weights('E:\\Code_v1\\forD2\\model4D2\\withuncertain_n10_weights_n\\')
    
    # results = model.predict_generator(testGene,582,verbose=1) #predict images in the Training pool
    # ## save the prediction
    # # saveResult('E:\\Code_v1\\mask_pool\\','E:\\Code_v1\\training_pool\\',results)
    # # batch_rgb2bianry('E:\\Code_v1\\mask_pool\\')
    # print('finished')



    
    
    
    
    



