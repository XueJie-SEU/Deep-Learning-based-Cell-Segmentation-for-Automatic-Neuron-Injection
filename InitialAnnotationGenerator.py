# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:18:56 2022

@author: user
"""
import os
import numpy as np
import umap
import scipy
import cv2
import scipy.ndimage as ndi
from PIL import Image
from matplotlib import pyplot as plt
from skimage import morphology,io,exposure
from skimage.filters.rank import  entropy, minimum, maximum, windowed_histogram, median
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



class initialAnnotationGenerator(object):
    def __init__(self):
       pass
   
    # function to display figure 
    def display_fig(self, num, fig, title = 'default'):
        plt.figure(num)
        plt.axis('off')
        io.imshow(fig)
    
    # local std
    def std_convoluted(self, image, N):
        im = np.array(image, dtype=float)
        im2 = im**2
        ones = np.ones(im.shape)
    
        kernel = np.ones((2*N+1, 2*N+1))
        s = scipy.signal.convolve2d(im, kernel, mode="same")
        s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
        ns = scipy.signal.convolve2d(ones, kernel, mode="same")
        return np.sqrt((s2 - s**2 / ns) / ns)
    
    # local variance
    def local_variance(self, image, N):
        im = np.array(image, dtype=float)
        im2 = im**2
        ones = np.ones(im.shape)  
        kernel = np.ones((2*N+1, 2*N+1))
        s = scipy.signal.convolve2d(im, kernel, mode="same")
        s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
        ns = scipy.signal.convolve2d(ones, kernel, mode="same")
        return (s2 - s**2 / ns) / ns
    
    # local uniformity
    def local_Uniformity(self, image, N):
        h, w = image.shape
        local_uniformity = np.zeros(image.shape)
        hist_img = windowed_histogram(image, morphology.square(N))
        for i in range(h):
            for j in range(w):
                local_uniformity[i,j] = sum(hist_img[i,j,:]**2)
        return local_uniformity
    
    # Get Data For Clustering    
    def GetDataForCluster(self, imagePath, savefolder = './featureMap1'):
        # create a folder if you want to save the feature map, then set savefolder='./featureMap1', or set the savefolder as False
        if savefolder:
            print('saveFolder:'+savefolder) 
            isExists = os.path.exists(savefolder)
            if not isExists:
                os.makedirs(savefolder)
                
        im_origin = io.imread(imagePath)
        h,w = im_origin.shape
        data = []
        n_rows = h*w
        # rescale the image intensity
        im_rescaled = exposure.rescale_intensity(im_origin)
        for kernel_size in range(5, 14, 2): #6-10
            if savefolder:
                if not os.path.exists(savefolder + '/ks' + str(kernel_size)):
                    os.makedirs(savefolder + '/ks' + str(kernel_size))
                
            N = kernel_size*2 + 1 
            name = imagePath.split('/')[-1].split('.')[0]
            
            # loc_var = self.local_variance(im_rescaled, kernel_size).reshape(n_rows, 1)
            # io.imsave(savefolder + '/ks' + str(kernel_size)+'/'+name+'_var.png',loc_var.reshape(h,w))
            # # self.display_fig(2,loc_var.reshape(h,w),'var')
            # data.append(loc_var)
            
            loc_uni = self.local_Uniformity(im_rescaled, N).reshape(n_rows, 1)
            loc_uni_rescaled = exposure.rescale_intensity(loc_uni)      
            if savefolder:
                print(savefolder)
                io.imsave(savefolder + '/ks' + str(kernel_size)+'/'+name+'_uni.png',loc_uni_rescaled.reshape(h,w))
                # self.display_fig(3,loc_uni_rescaled.reshape(h,w),'loc_uni')
            data.append(loc_uni_rescaled)
            
            loc_range = (maximum(im_rescaled, morphology.square(N))-minimum(im_rescaled, morphology.square(N))).reshape(n_rows, 1)
            if savefolder:
                io.imsave(savefolder + '/ks' + str(kernel_size)+'/'+name+'_ran.png',loc_range.reshape(h,w))
                # self.display_fig(4,loc_range.reshape(h,w),'loc_range')
            data.append(loc_range)
            
            loc_medium = median(im_rescaled,  morphology.square(kernel_size)).reshape(n_rows, 1)
            if savefolder:
                io.imsave(savefolder + '/ks' + str(kernel_size)+'/'+name+'_med.png',loc_medium.reshape(h,w))
                # self.display_fig(5,loc_medium.reshape(h,w),'loc_medium')
            data.append(loc_medium)
            
            loc_entropy = entropy(im_rescaled, morphology.square(N)).reshape(n_rows, 1)
            if savefolder:
                io.imsave(savefolder + '/ks' + str(kernel_size)+'/'+name+'_ent.png',loc_entropy.reshape(h,w))
                # self.display_fig(6,loc_entropy.reshape(h,w),'loc_entropy')
            data.append(loc_entropy)
    
        data = np.column_stack(data)    
        print('1 image finished!')    
        return np.mat(data),h,w
    
    
    def clusteringAfterUmapOrPCA(self, imgFeaData, row, col, n_clusters, decoMethod = 'UMAP'):
        # normalization
        scaler = StandardScaler().fit(imgFeaData)
        imgData_scl = scaler.transform(imgFeaData)  
        
        if decoMethod == 'PCA':
            print('Here PCA')
            # # pca
            pca = PCA(n_components=3)
            clusterable_embedding = pca.fit_transform(imgData_scl)
        else: 
            #umap
            print('Here UMAP')
            clusterable_embedding = umap.UMAP(
                n_neighbors=60,
                min_dist=0.0,
                n_components=3,
                random_state=None,
            ).fit_transform(imgData_scl)    
        # KMeans clustering    
        km = KMeans(n_clusters)
        label1 = km.fit_predict(clusterable_embedding).reshape([row,col]) 
        io.imsave('./Example/K_'+str(n_clusters)+'_'+decoMethod+'.png', label1)
        self.display_fig(7,label1,'im_label')
    
    
     # label for M2
    def M2_unknown(self, roughMask):
        bk_temp = morphology.dilation(roughMask, selem=morphology.disk(10))
        unknown = bk_temp - roughMask
        bk_temp[unknown==255]=128
        labelM2 = Image.fromarray(bk_temp)
        labelM2.save('./Example/label_M2.png')
        
        
    # label for M1
    def M1_watershed(self, roughMask, img):
        bk_temp = morphology.dilation(roughMask, selem=morphology.disk(10))
        unknown = bk_temp - roughMask
        ret, markers = cv2.connectedComponents(roughMask)
        markers = markers+1
        markers[unknown==255]=0
        markers = cv2.watershed(img, markers)
        img[markers == 2] = [255,255,255]
        labelM1 = Image.fromarray(markers==2)
        labelM1.save('./Example/label_M1.png')
        
   # label for M3
    def M3_watershed_uncertain(self, M1Mask):
       dilation_t = morphology.dilation(M1Mask, selem=morphology.disk(7))
       erosion_t = morphology.erosion(M1Mask, selem=morphology.disk(3))
       unknown = dilation_t - erosion_t
       dilation_t[unknown==255]=128
       self.display_fig(10,dilation_t,'label_M3')
       label_ori = Image.fromarray(dilation_t)
       label_ori.save('./Example/label_M3.png')

  
            

if __name__ == '__main__':
    
    Generator1 = initialAnnotationGenerator()
    d1, h, w = Generator1.GetDataForCluster('./Example/632.tiff', False)
    Generator1.clusteringAfterUmapOrPCA(d1, h, w, 4)
    
    ##****adjust the label****##
    label = io.imread('./Example/K_4_UMAP.png')
    roughMask = (label==0)
    roughMask = morphology.remove_small_objects(np.array(roughMask,bool),min_size=900,connectivity=1)*255
    roughMask_arr = ndi.binary_fill_holes(roughMask, structure=np.ones((3,3)))
    
    # M2 label
    Generator1.M2_unknown(np.uint8(roughMask_arr*255))
    
    # save rough mask
    roughMask = Image.fromarray(roughMask_arr)
    roughMask.save('./Example/632_roughMask.png')
    ##************************##
    
    
    # M1 label
    image = cv2.imread('./Example/632.tiff')
    Generator1.M1_watershed(np.uint8(roughMask_arr*255),image)
    
    # M3 label
    M1Mask = io.imread('./Example/label_M1.png')
    Generator1.M3_watershed_uncertain(M1Mask)

    