# coding:utf-8
import cv2
import dlib
from scipy.misc import imresize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    data_path = "E:/mySuperResolution/dataset/Set5/"
    image_name = "butterfly_GT.bmp"

    rgb = cv2.imread(data_path+image_name)
    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)[:,:,0]
    img = img[0:img.shape[0] - img.shape[0] % 3 ,0:img.shape[1] - img.shape[1] % 3 ]
    s = img.shape
    print s
    img1 = imresize(img,(s[0]/3, s[1]/3))
    img1r = imresize(img1,(s[0], s[1]),interp='bicubic')

    plt.subplot(521)
    plt.imshow(img1r,cmap='gray')
    plt.subplot(522)
    resedul = img - img1r
    plt.imshow(np.power(resedul,2),cmap='gray')

    img2 = imresize(img1r,(s[0]/3, s[1]/3))
    img2r = imresize(img2,(s[0], s[1]),interp='bicubic')

    plt.subplot(523)
    plt.imshow(img2r,cmap='gray')
    plt.subplot(524)
    resedul = img1r - img2r
    plt.imshow(np.power(resedul,2),cmap='gray')

    img3 = imresize(img2r,(s[0]/3, s[1]/3))
    img3r = imresize(img3,(s[0], s[1]),interp='bicubic')


    plt.subplot(525)
    plt.imshow(img3r,cmap='gray')
    plt.subplot(526)
    resedul = img2r - img3r
    plt.imshow(np.power(resedul,2),cmap='gray')

    img4 = imresize(img3r,(s[0]/3, s[1]/3))
    img4r = imresize(img4,(s[0], s[1]),interp='bicubic')


    plt.subplot(527)
    plt.imshow(img4r,cmap='gray')
    plt.subplot(528)
    resedul = img3r - img4r
    plt.imshow(np.power(resedul,2),cmap='gray')

    img5 = imresize(img4r,(s[0]/3, s[1]/3))
    img5r = imresize(img5,(s[0], s[1]),interp='bicubic')

    plt.subplot(529)
    plt.imshow(img5r,cmap='gray')
    plt.subplot(5,2,10)
    resedul = img4r - img5r
    plt.imshow(np.power(resedul,2),cmap='gray')
    plt.show()

    for i in range(100):
        tmp = np.copy(img5r.copy())
        img5 = imresize(img5r,(s[0]/3, s[1]/3))
        img5r = imresize(img5,(s[0], s[1]),interp='bicubic')

        plt.subplot(131)
        plt.imshow(img5r,cmap='gray')
        plt.subplot(132)
        resedul = tmp - img5r
        plt.imshow(np.power(resedul,2),cmap='gray')
        plt.subplot(133)
        resedul = img - img5r
        plt.imshow(np.power(resedul,2))
        plt.show()