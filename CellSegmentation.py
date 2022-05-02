# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:18:51 2021

@author: user
"""
import numpy as np
import os
from skimage import io
from attUNet4Mask import att_unet
from skimage import morphology
from tensorflow.keras.models import *


def predict_fun(dir_str):
    images = os.listdir(dir_str)    

    # load the model
    model = att_unet()
    # model.load_weights('.\\model\\M1\\watershed_n9_weights\\') # M1
    # model.load_weights('.\\model\\M2\\withuncertain_n10_weights_n\\') # M2
    model.load_weights('.\\model\\M3\\watershed_uncertain_n10_weights\\') # M3
    
    for filename in images:
        prefix = os.path.splitext(filename)[1]
        if prefix in ['.tiff' , '.jpg']:    
            im_name = os.path.splitext(filename)[0]
            img_origin = io.imread(dir_str+filename)
            img_origin = ((img_origin-img_origin.min()) / (img_origin.max()-img_origin.min()))*255
            img_origin = img_origin/255.
            pred = model.predict(np.reshape(img_origin,(1,)+img_origin.shape+(1,)))
            pred = pred[0,:,:,0]
            thresh = 0.5
            img = morphology.remove_small_objects(np.array((pred >= thresh),bool),min_size=2400,connectivity=1)
            img = np.uint8(img*255) #根据阈值进行分割
            io.imsave('.\\model\\predRes\\'+im_name+'_M3.png',img)
    print('finished')
    
   
 

if __name__ == '__main__':
    img_root = '.\\model\\img4test\\'
    predict_fun(img_root)
    
