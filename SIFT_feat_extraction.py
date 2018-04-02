import cv2
import numpy as np
import pandas as pd
import os
#import sklearn
#from sklearn.cluster import KMeans
import pickle


def saveFeaturesBOW():
    files = os.listdir("/Users/phaniram/Downloads/AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]")
    dic = {}
    for file in files:
        filename = '/Users/phaniram/Downloads/AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/' + file
        im= cv2.imread(filename)
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fea_det=cv2.FeatureDetector_create("SIFT")
        des_ext=cv2.DescriptorExtractor_create("SIFT")
        kpts = fea_det.detect(gray_im)
        #print(len(kpts))
        #img2 = cv2.drawKeypoints(gray_im, kpts, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kpts, desc = des_ext.compute(gray_im, kpts)
        #print(len(kpts))
        #print((desc[0]))
        for i in range(0, len(kpts)):
            normalizingValue = np.sum(desc[i])
            desc[i] /= (normalizingValue)
        #normalizingValue = np.sum(desc[0])
        #desc[0] /= (normalizingValue)
        #print(desc[0])
        if(len(kpts)>=20):
            #dic.add({file: desc})
            dic[file] = desc
    with open('features.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


def minMaxFeaturesBOW():
    files = os.listdir("/Users/phaniram/Downloads/AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]")
    min = 10000
    max =0
    sum = 0
    count =0
    non_outliers = 0
    for file in files:
        filename = '/Users/phaniram/Downloads/AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/' + file
        im= cv2.imread(filename)
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fea_det=cv2.FeatureDetector_create("SIFT")
        des_ext=cv2.DescriptorExtractor_create("SIFT")
        kpts = fea_det.detect(gray_im)
        sum = sum+ len(kpts)
        count = count +1
        if(len(kpts)>20):
            non_outliers = non_outliers+1
        if(len(kpts)>max):
            max = len(kpts)
        if(len(kpts)<min):
            min = len(kpts)

    print(non_outliers)
    print(count)
    print(sum/count)
    print(min)
    print(max)

saveFeaturesBOW()
#minMaxFeaturesBOW()