# -*- coding: utf-8 -*-
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.cm as cm
import time

from scipy.signal import convolve2d
from sklearn.decomposition import SparseCoder
from sklearn.cluster import KMeans
import toolgate.deformed as Deformed
import math
import toolgate.colormanage as tc


from scipy.misc import imresize

import skimage.io as io
__author__ = 'liangz14'


def sc_result_analysis():
    """
        对稀疏编码的结果进行分析i
    :return:
    """
    sc_file = open('./tmp_file/30_dictionary.pickle', 'rb')
    sc_list = cPickle.load(sc_file)

    classified_file = open('./tmp_file/30_class_result.pickle', 'rb')
    (classified_feature, classified_patch) = cPickle.load(classified_file)

    model_file = open('./tmp_file/30_kmeans_pca_model.pickle', 'rb')
    (k_means, pca) = cPickle.load(model_file)

    sc_file.close()
    classified_file.close()
    model_file.close()

    # ========================================================================
    for i in range(5):
        k = i

        #v_feature = pca.transform(classified_feature[1][k]).reshape((-1,))
        v_feature = classified_feature[3][k]
        v_patch = classified_patch[3][k]

        feature_dict = sc_list[0][:, :144]
        patch_dict = sc_list[0][:, 144:]

        #v_feature = feature_dict[0]
        #v_patch = patch_dict[0]

        coder = SparseCoder(dictionary=feature_dict, transform_algorithm='omp',
                            transform_alpha=0.01, n_jobs=2, transform_n_nonzero_coefs=1)

        weight = coder.transform(v_feature)

        v_patch = v_patch.reshape((9, 9))
        result = np.dot(weight, patch_dict).reshape((9, 9))

        mask = weight != 0
        print weight[mask]
        mask = mask[0]

        print len(patch_dict[mask])
        print len(patch_dict[mask])
        patch_show(patch_dict[mask],[0,0,0.45,0.45],1)

        ax2 = plt.axes([0, 0.5, 0.45, 0.45])
        ax2.imshow(result, interpolation="none", cmap=cm.gray)

        ax2 = plt.axes([0.5, 0.5, 0.45, 0.45])
        ax2.imshow(v_patch, interpolation="none", cmap=cm.gray)

        plt.show()


def patch_class_show():

    # ========================读取经过切分的训练数据=============================
    classified_file = open('./tmp_file/2_class_with_regression.pickle', 'rb')
    #classified_file = open('./tmp_file/1_class_result.pickle', 'rb')
    (classified_feature, classified_patch,_) = cPickle.load(classified_file)
    classified_file.close()

    num = []
    for i in classified_feature:
        num.append(len(i))
    print num

    for i in range(9):
        x = i % 3
        y = i / 3

        patch_show(classified_patch[i+5][:625], [0.05+x*0.31, 0.05+y*0.31, 0.3, 0.3], i)
    plt.show()


def patch_single_class_show():

    for i in range(50):
        # ========================读取经过切分的训练数据=============================
        # classified_file = open('./tmp_file/6_class_result.pickle', 'rb')
        classified_file = open('./tmp_file/_tmp_sc_classify_data.pickle', 'rb')
        classified_feature,classified_patch, noise_feature, noise_patch,error_mean = cPickle.load(classified_file)
        classified_file.close()

        patch_show(classified_patch[i][:10000], [0.05, 0.05, 0.9, 0.9], 1)
        plt.show()


def dict_class_show():
    """
        直接处理字典分类
    """
    sc_file = open('./tmp_file/9_dictionary_regression.pickle', 'rb')
    sc_list = cPickle.load(sc_file)

    for i in sc_list:
        print len(i)

    for i in range(9):
        x = i % 3
        y = i / 3
        # 进行一次系数编码测试
        q = i
        Dl_dict = sc_list[q][:, :144].T
        Dh_dict = sc_list[q][:, 144:].T
        patch_show(Dh_dict.T[:100], [0.05+x*0.31, 0.05+y*0.31, 0.3, 0.3], i)
    plt.show()


def dict_single_class_show():
    for i in range(20):
        #sc_file = open('./tmp_file/11_dictionary_regression.pickle', 'rb')
        sc_file = open('./tmp_file/_tmp_sc_list_new_clsd_raw_14.pickle', 'rb')

        sc_list = cPickle.load(sc_file)
        eps = np.finfo(float).eps

        Dh_dict = sc_list[i][:, 30:]
        Dl_dict = sc_list[i][:, :30]

        mes = (np.sum(Dl_dict ** 2, axis=1)) ** 0.5 + eps
        print len(mes)
        mask = mes > 0.6

        patch_show(Dh_dict, [0.05, 0.05, 0.9, 0.9], 1)
        plt.show()


def patch_show(patch_list, c, s):
    """
        显示提取出来的patch
        patch:(patch num, pixels num)
        cL:color
        s:tag
    """
    ax = plt.axes(c)
    ax.text(0, 0, s)
    l = int(len(patch_list) ** 0.5)
    l_in = int(len(patch_list[0]) ** 0.5)

    img = np.zeros((l * (l_in + 3) + 3, l * (l_in + 3) + 3))
    # print img.shape
    step = l_in + 3
    for index in range(l * l):
        y = index / l
        x = index % l

        p = patch_list[index].reshape((l_in, l_in))
        p = onto_unit(p)
        img[3 + y * step:3 + y * step + l_in, 3 + x * step:3 + x * step + l_in] = p
    ax.axis('off')

    ax.imshow(img, interpolation="none", cmap=cm.gray)


def onto_unit(x):
    """
    对patch进行统一范围
    """
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)


def patch_classify():
    """
        patch可视化：观察patch在。
        PCA空间，训练数据和实际数据的关系。
        构造了kd-tree
    """
    with open('training_data_full.pickle') as f:
        # 读取对应的原始patch
        kk = open("raw_data_full.pickle", 'rb')
        raw_lib = cPickle.load(kk)
        raw_lib = np.asarray(raw_lib, dtype='float32')

        # 读取数据转换特征
        training_data = cPickle.load(f)
        patch_lib, feature_lib = training_data
        feature_lib, patch_lib = (np.asarray(feature_lib, dtype='float32'), np.asarray(patch_lib, dtype='float32'))
        feature_lib = feature_lib.reshape((-1, 4 * 9 * 9))

        # 构造KD-tree
        tree = KDTree(feature_lib, leaf_size=len(feature_lib) / 100)

        # 在KD-tree当中搜索最近的100个点
        dist, ind1 = tree.query(feature_lib[5678], k=100)
        nn1 = feature_lib[ind1][0]

        dist, ind2 = tree.query(feature_lib[10000], k=100)
        nn2 = feature_lib[ind2][0]

        dist, ind3 = tree.query(feature_lib[1233], k=100)
        nn3 = feature_lib[ind3][0]

        # 计算并转换PCA空间
        pca = PCA(n_components=2)
        d2_data = pca.fit_transform(feature_lib).T

        # 降临近点的高维坐标转换成PCA空间的低维坐标
        r1 = pca.transform(nn1).T
        r2 = pca.transform(nn2).T
        r3 = pca.transform(nn3).T

        # 设置绘制范围
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])

        # 绘制全部数据的散点图
        ax.scatter(d2_data[0], d2_data[1], c='g')
        # 绘制三个类别的散点图
        ax.scatter(r1[0], r1[1], c='r')
        ax.scatter(r2[0], r2[1], c='b')
        ax.scatter(r3[0], r3[1], c='y')

        # patch_lib \ raw_lib分别是差值patch和原始patch
        patch_show(raw_lib[ind1][0], [0.05, 0.05, 0.4, 0.4], 'red')
        patch_show(raw_lib[ind2][0], [0.05, 0.55, 0.4, 0.4], 'blue')
        patch_show(raw_lib[ind3][0], [0.55, 0.05, 0.4, 0.4], 'yellow')

        plt.show()


def k_means_classify(data_list, n_clusters=15, n_components=30, pca=None):
    """
        使用k-mean对patch进行分类
        list 原始数据 （num，dim）
        :n_clusters: 需要分类的数量
        :n_components: 需要使用的维度
        :return: 表示分类结果
    """
    if pca is None:
        # 将原本的数据进行降维
        pca = PCA(n_components=n_components)
        pca = pca.fit(data_list)
        data_list = pca.transform(data_list)
    else:
        pass

    # 进行k-means聚类
    k_means = KMeans(n_clusters=n_clusters)
    k_means = k_means.fit(data_list)
    y_predict = k_means.predict(data_list)

    # return y_predict
    return y_predict, k_means, pca


def re_classify_dict():
    dict_file = open("_dictionary.pickle", "rb")
    sc_list = cPickle.load(dict_file)
    sc_list = np.concatenate(sc_list)

    Dh_dict = sc_list[:, 144:]
    Dl_dict = sc_list[:, :144]

    k_means = KMeans(n_clusters=15)
    k_means = k_means.fit(Dl_dict)
    y_predict = k_means.predict(Dl_dict)

    num = []
    y_tmp = np.asarray(y_predict, dtype=int) * 0 + 1
    for i in range(len(np.unique(y_predict))):
        num.append(np.sum(y_tmp[y_predict == i]))
    rand = np.asarray(num).argsort()  # 按照各个类别patch个数从少到多排序的类别索引

    classified_hdict = []
    classified_patch = []
    for i in rand:
        predict_temp = y_predict == i
        classified_hdict.append(Dh_dict[predict_temp])
        print len(classified_hdict[-1])

    for i in range(9):
        x = i % 3
        y = i / 3
        # 进行一次系数编码测试
        patch_show(classified_hdict[i+5][:100], [0.05+x*0.31, 0.05+y*0.31, 0.3, 0.3], i)

    plt.show()


def deformed_test():

    # 读入变形起始的参数
    f = open("deformed_source.pickle", 'rb')
    lim_patch_list, rim_patch_list, image_patch, patch_list = cPickle.load(f)
    f.close()

    deformed_list = []
    print np.mean((np.asarray(image_patch)-np.asarray(patch_list))**2)**0.5

    fx = np.asarray([[1.0, -1.0]], dtype='float')
    fy = np.asarray([[1.0], [-1.0]], dtype='float')

    for i in range(len(lim_patch_list)):
        grad_x = convolve2d(patch_list[i], fx, mode='same')*0.1
        grad_y = convolve2d(patch_list[i], fy, mode='same')*0.1

        grad_x[:, 0] = grad_x[:, 0]*0+1
        grad_y[0, :] = grad_y[0, :]*0+1

        # ====================== 这里手动构造梯度 ===================

        Def = Deformed.deformed_patch()
        cp, c = Def.deform(patch_list[i], image_patch[i],grad_x,grad_y)
        deformed_list.append(cp)
        print np.mean((patch_list[i]-image_patch[i])**2)**0.5
        print np.mean((cp-image_patch[i])**2)**0.5
        print

        plt.subplot(221)
        plt.imshow(rim_patch_list[i], cmap=cm.gray, interpolation="none")
        plt.subplot(222)
        plt.imshow(patch_list[i], cmap=cm.gray, interpolation="none")
        plt.subplot(223)
        plt.imshow(cp-np.min(cp), cmap=cm.gray, interpolation="none")
        plt.subplot(224)
        plt.imshow(image_patch[i], cmap=cm.gray, interpolation="none")
        plt.show()

    print "finished"
    print np.mean((np.asarray(deformed_list)-np.asarray(patch_list))**2)**0.5

def deformed():
    # 读入变形起始的参数
    f = open("deformed_source.pickle", 'rb')
    lim_patch_list, rim_patch_list, patch_list = cPickle.load(f)
    f.close()

    for i in range(len(lim_patch_list)):
        Def = Deformed.deformed_patch()
        (u, v, cp) = Def.getUV(rim_patch_list[i], patch_list[i])

        u = u.reshape(u.shape[2], u.shape[3])
        v = v.reshape(v.shape[2], v.shape[3])
        cp = cp.reshape(cp.shape[2], cp.shape[3])
        plt.subplot(2, 3, 1)
        plt.title("x opp")
        plt.imshow(v, cmap=cm.gray, interpolation="none")
        plt.subplot(2, 3, 4)
        plt.title("y opp")
        plt.imshow(u, cmap=cm.gray, interpolation="none")
        plt.subplot(2, 3, 2)
        plt.title("input")
        plt.imshow(rim_patch_list[i], cmap=cm.gray, interpolation="none")
        plt.subplot(2, 3, 3)
        plt.title("aim")
        plt.imshow(patch_list[i], cmap=cm.gray, interpolation="none")
        plt.subplot(2, 3, 5)
        plt.title("output")
        plt.imshow(cp, cmap=cm.gray, interpolation="none")
        plt.subplot(2, 3, 6)
        plt.title("lim")
        plt.imshow(lim_patch_list[i], cmap=cm.gray, interpolation="none")
        plt.show()

    pass

def combine():
    im_my = io.imread('baby_MY[gray]sc.bmp')[3:507, 3:507]
    origin = io.imread('baby_GT[gray].bmp')
    im_jor = io.imread('baby_JOR[gray].bmp')

    diff_my = np.abs(origin-im_my)
    diff_jor = np.abs(origin-im_jor)
    mask = 1.0*diff_my > 1.1*diff_jor

    result = im_my.copy()
    result[mask] = im_jor[mask]

    print psnr(origin, result)

    plt.imshow(result, interpolation="none", cmap=cm.gray)
    plt.show()
    pass


def test_bicubic():
    origin = io.imread('baby_GT[gray].bmp')
    im_jor = io.imread('baby_JOR[gray].bmp')
    im_my = io.imread("baby_MY[gray].bmp")

    image = io.imread('baby_GT.bmp')
    shape = origin.shape

    if len(shape) == 3:
        test_img = image[:shape[0]-shape[0] % 3, :shape[1]-shape[1] % 3, :]
    else:
        test_img = image[:shape[0]-shape[0] % 3, :shape[1]-shape[1] % 3]

    lim = imresize(test_img, 1/3.0, 'bicubic')
    mim = imresize(lim, 2.0, 'bicubic')
    rim = imresize(lim, 3.0, 'bicubic')

    lim = np.asarray(tc.rgb2ycbcr(lim)[:, :, 0], dtype=float)
    image = np.asarray(tc.rgb2ycbcr(test_img)[:, :, 0], dtype=float)
    mim = np.asarray(tc.rgb2ycbcr(mim)[:, :, 0], dtype=float)
    rim = np.asarray(tc.rgb2ycbcr(rim)[:, :, 0], dtype=float)

    print psnr(image*1.0, rim*1.0)
    print psnr(image*1.0, im_my[0:504,0:504]*1.0)

    plt.subplot(221)
    plt.imshow(image, interpolation="None", cmap=cm.gray)
    plt.subplot(222)
    plt.imshow(np.abs(rim), cmap=cm.gray)
    plt.subplot(223)
    plt.imshow(np.abs(im_my), interpolation="None", cmap=cm.gray)
    plt.subplot(224)
    plt.imshow(np.abs(im_jor), interpolation="None", cmap=cm.gray)

    plt.show()


def psnr(img1, img2):
    """
    psnr计算过程数据范围需要在 0 - 255 之间，并且需要是二维矩阵
    :param img1:
    :param img2:
    :return:
    """
    mse = np.sum((img1/255.0 - img2/255.0) ** 2)
    # print img1
    if mse == 0:
        return 100
    N = img1.shape[1] * img1.shape[0]
    return 10 * math.log10(N / mse)


def psnr_test():
    """
    测试JOR算法生成的两张图片的psnr值

    :return:
    """
    im_origin = io.imread("baby_GT[1-Original].bmp")
    im_jor = io.imread("baby_GT[12-JOR].bmp")
    print np.max(im_jor)
    print np.max(im_origin)
    print psnr(im_origin, im_jor)

#combine()
#deformed()
#deformed_test()
#psnr_test()
#test_bicubic()
#patch_class_show()
patch_single_class_show()
#dict_single_class_show()
#sc_result_analysis()
# patch_class_show()
# sc_result_analysis()
# sc_result_analysis2()

#no pca