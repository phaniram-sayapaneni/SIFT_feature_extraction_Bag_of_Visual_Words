import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

filename = '/Users/phaniram/Downloads/AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/'
filename = filename+ '0002b_num2.png'
im= cv2.imread(filename)
cv2.imshow('img', im)
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(cv2.__version__)
fea_det=cv2.FeatureDetector_create("SIFT")
des_ext=cv2.DescriptorExtractor_create("SIFT")

print(fea_det)
kpts = fea_det.detect(gray_im)
print(len(kpts))
img2 = cv2.drawKeypoints(gray_im, kpts, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img2)
plt.show()
