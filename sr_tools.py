# coding:utf-8
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imresize
from scipy.signal import convolve2d
import numpy as np
import cPickle
import math
import toolgate.deformed as Deformed
from sklearn.neighbors import KDTree
import skimage.filters as ft
import time
from toolgate.colormanage import rgb2ycbcr,ycbcr2rgb
from sklearn import linear_model
import cv2
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from toolgate import gabor
import random
from toolgate.superpixel import SLIC
__author__ = 'liangz14'


def image_pre_filter(img):
    """
    对输入的单通道图像进行预处理：
    包括中值滤波，锐化，降噪等操作，办证patch的效果
    :param img:
    :return:
    """
    #plt.subplot(121)
    #plt.imshow(img, interpolation="none", cmap=cm.gray)

    gs = ft.gaussian_filter(img, 0.9)
    result = img + (img - gs)*4

    result[result > 255] = 255
    result[result < 0] = 0
    #plt.subplot(122)
    #plt.imshow(result, interpolation="none", cmap=cm.gray)

    #plt.show()
    return result


def patch_show(patch_list, c, s):
    """
        显示提取出来的patch
        patch:(patch num, pixels num)
        cL:color
        s:tag
    """
    #print len(patch_list)
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


def mode(l):
    """
    求众数
    :param l:
    :return:
    """
    inde = [np.argsort([l.count(i) for i in l])]
    k = np.asarray(l)
    return k[inde][-1]


def heavy_feature(img):
    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')

    feature = np.zeros((2, img.shape[0], img.shape[1]))
    feature[0, :, :] = convolve2d(img, f3, mode='same')
    feature[1, :, :] = convolve2d(img, f4, mode='same')
    grad = feature[0, :, :] + feature[1, :, :]
    l_grad = grad.reshape((grad.shape[0] * grad.shape[1],))
    hist, bin_edges = np.histogram(l_grad, bins=200)
    hist = list(np.log2(hist*1.0 / np.sum(hist)))
    return hist, bin_edges[:-1]


def get_patch_class(feature_list, pca=None, kmeans=None):
    """
    通过分析feature_list得到patch所属的分类
    :param feature_list:  特征列表
    :param pca: pca降维对象 可以直接调用fit
    :param kmeans: kmean聚类对象 可以直接调用predict
    :return:
    """
    orig_feature_list = feature_list.copy()
    eps = np.finfo(float).eps
    # X除均方差
    mes = (np.sum(feature_list ** 2, axis=1)) ** 0.5+eps

    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_list = feature_list / mes_list

    if pca is not None:
        feature_list = pca.transform(feature_list)

    class_list = []
    if kmeans is not None:
        class_list = kmeans.predict(feature_list)
    else:
        class_list = np.asarray(np.zeros((np.shape(feature_list)[0],)), dtype=int)
    '''
    plt.hist(class_list, bins=14)
    plt.show()
    '''
    class_list = np.asarray(np.zeros((np.shape(feature_list)[0],)), dtype=int)
    return class_list, orig_feature_list


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


def read_super_patch(file_name="tmp_file.pickle"):
    """
    从文件读取已经计算出结果的pathclist
    :return:
    """
    # k = open("tmp_result_3times.pickle", 'rb')
    k = open(file_name, 'rb')

    result = cPickle.load(k)
    k.close()

    return result


def super_pixel_classify(image, cls_num=30):
    nr_superpixels = cls_num
    nc = int(17)
    step = int((image.shape[0]*image.shape[1]/nr_superpixels)**0.5)
    slic = SLIC(image, step, nc)
    slic.generateSuperPixels()
    slic.createConnectivity()
    return slic.clusters


def get_super_patch_sc(feature_list, patch_list=None, tag=1):
    """
    得到得到高分辨率的patch
    patch_list:需要进行超分辨率的patch
    class_list:该patch所在类别

    :return:
    """
    eps = np.finfo(float).eps
    # ===========================  加载字典   ===========================
    sc_file = open('./tmp_file/%d_dictionary_regression.pickle' % tag, 'rb')
    sc_list = cPickle.load(sc_file)
    sc_file.close()

    sum_list = []
    # ======================== 对字典进行过滤==========================
    for i in range(len(sc_list)):
        filter_list = sc_list[i][:, :30]
        mes = (np.sum(filter_list ** 2, axis=1)) ** 0.5 + eps
        mask = mes > 0.3
        sum_list.extend(sc_list[i][mask])
        sc_list[i] = sc_list[i][mask]
    # ==========================================================

    # =========================== 加载回归模型以及分类好的数据 ===================
    classified_file = open('./tmp_file/%d_class_with_regression.pickle' % tag, 'rb')
    classified_feature, classified_patch, classified_error, class_tag = cPickle.load(classified_file)
    classified_file.close()

    # ===========================  读取模型：pca以及kmeans ===============
    model_file = open('./tmp_file/%d_kmeans_pca_model.pickle' % tag, 'rb')
    (k_means, pca) = cPickle.load(model_file)
    model_file.close()

    # ===========================  进行测试patch分类 ====================
    class_list, feature_list = get_patch_class(feature_list, pca, k_means)
    feature_list = pca.transform(feature_list)

    class_tag_dic = []
    for i in range(len(sc_list)):
        class_tag_dic.append(np.zeros((len(sc_list[i]),))+i)

    dict = np.asarray(np.concatenate(sc_list, axis=0))
    classified_feature = np.asarray(np.concatenate(classified_feature, axis=0))  # 特征的集合
    class_tag = np.asarray(np.concatenate(class_tag, axis=0)) #训练数据的集合
    class_tag_dic = np.asarray(np.concatenate(class_tag_dic, axis=0)) #字典数据的集合

    # X除均方差
    fea_mean = np.mean(feature_list, axis=1)
    fea_mean = np.repeat([fea_mean], len(feature_list[1]), axis=0).T
    feature_list = (feature_list - fea_mean)

    mes = (np.sum(feature_list ** 2, axis=1)) ** 0.5 + eps
    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_knn = 1.0*feature_list / mes_list

    dict = dict[:, :30]
    dic_l_tree = KDTree(dict, leaf_size=len(dict)/2)
    orig_l_tree = KDTree(classified_feature, leaf_size=len(classified_feature) / 100)

    dist4, ind4 = dic_l_tree.query(feature_knn[:], k=16)

    # ===========================  对分好类的所有patch进行分辨率提升 =======
    result = []
    cla = []
    for ind, fea in enumerate(feature_list):

        # ======================== 读取某个分类下的字典 ==============================
        most = np.argsort(dist4[ind])
        lost = class_tag_dic[ind4[ind]]
        patch_class = np.int(mode(lost[most].tolist()))

        cla.append(patch_class)
        Dl_dict = sc_list[patch_class][:, :30]
        Dh_dict = sc_list[patch_class][:, 30:]
        #Dl_dict = np.asarray(sum_list)[:, :30]
        #Dh_dict = np.asarray(sum_list)[:, 30:]
        normalization_m = np.sqrt(np.sum(fea**2)) + eps

        yy = fea/normalization_m
        '''
        # ======================== 使用稀疏编码的对象求解 =========================
        sp = SparseCoder(dictionary=Dl_dict, transform_algorithm='lasso_lars',
                   transform_alpha=0.001, transform_n_nonzero_coefs=3, n_jobs=2)

        w = sp.transform(yy)
        '''
        # ======================== 直接使用LASSO对象求解 =========================
        clf = linear_model.Lasso(alpha=0.00003)
        clf.fit(Dl_dict.T, yy)
        w = clf.coef_

        tmp_result = np.dot(w, Dh_dict)*normalization_m

        result.append(tmp_result)

        print "已完成%0.2f %%" % (ind*1.0/len(feature_list)*100)

    # plt.hist(cla,bins=15)
    # plt.show()
    k = open("tmp_result.pickle", 'wb')
    cPickle.dump(result, k, 1)
    k.close()
    return result


def get_super_patch_regression_with_search(feature_list, patch_list=None, input_tag='1',model_tag='20',file_name="tmp_file.pickle"):
    """
    得到得到高分辨率的patch
    patch_list:需要进行超分辨率的patch
    class_list:该patch所在类别
    :return:
    """
    eps = np.finfo(float).eps

    # =========================== 加载回归模型以及分类好的数据 ===================
    classified_file = open('./tmp_file/%s_class_with_regression.pickle' % input_tag, 'rb')
    classified_feature, classified_patch, classified_error, class_tag = cPickle.load(classified_file)
    classified_file.close()

    regression_file = open('./tmp_file/%s_regression_ridge.pickle' % input_tag, 'rb')
    regression_list = cPickle.load(regression_file)
    regression_file.close()

    # ===========================  读取模型：pca以及kmeans ===============
    model_file = open('./tmp_file/%s_kmeans_pca_model.pickle' % model_tag, 'rb')
    (k_means, pca) = cPickle.load(model_file)
    model_file.close()

    # ===========================  进行测试patch分类 ====================
    class_list, feature_list = get_patch_class(feature_list, pca, k_means)
    feature_list = pca.transform(feature_list)
    # ===========================  对分好类的所有patch进行分辨率提升 =======
    result = []

    a = time.time()
    print "开始构造KD树\n"

    # 将特征进行合并
    classified_feature = np.asarray(np.concatenate(classified_feature, axis=0))  # 特征的集合
    class_tag = np.asarray(np.concatenate(class_tag, axis=0)) #训练数据的集合

    orig_l_tree = KDTree(classified_feature, leaf_size=len(classified_feature) / 100)

    b = time.time()
    print "KD树构造完成：耗时%f秒\n" % (b - a)
    a = b

    # X除均方差
    mes = (np.sum(feature_list ** 2, axis=1)) ** 0.5 + eps
    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_knn = 1.0*feature_list / mes_list
    dist4, ind4 = orig_l_tree.query(feature_knn[:], k=16)  #16近邻885秒； 4近邻500秒

    b = time.time()
    print "搜索完成：耗时%f秒\n" % (b - a)
    a = b

    for ind in range(len(feature_list)):
        most = np.argsort(dist4[ind])
        lost = class_tag[ind4[ind]]
        patch_class = mode(lost[most].tolist())

        regressor = regression_list[np.int(patch_class)]

        res = np.dot(regressor, feature_list[ind])

        result.append(res)

    b = time.time()
    print "计算完成：耗时%f秒\n" % (b - a)
    a = b

    k = open(file_name, 'wb')
    cPickle.dump(result, k, 1)
    k.close()
    return result


def get_super_patch_regression_mul_with_search(feature_list, patch_list=None, tag="mullayer36"):
    """
    得到得到高分辨率的patch
    patch_list:需要进行超分辨率的patch
    class_list:该patch所在类别
    :return:
    """
    eps = np.finfo(float).eps

    # =========================== 加载回归模型以及分类好的数据 ===================
    classified_file = open('./tmp_file/%s_class_with_regression.pickle' % tag, 'rb')
    classified_feature, classified_patch, classified_error, class_tag = cPickle.load(classified_file)
    classified_file.close()

    regression_file = open('./tmp_file/%s_regression_ridge.pickle' % tag, 'rb')
    regression_list = cPickle.load(regression_file)
    regression_file.close()

    # ===========================  读取模型：pca以及kmeans ===============
    model_file = open('./tmp_file/%s_kmeans_pca_model.pickle' % tag, 'rb')
    (k_means, pca) = cPickle.load(model_file)
    model_file.close()

    # ===========================  进行测试patch分类 ====================
    class_list, feature_list = get_patch_class(feature_list, pca, k_means)
    feature_list = pca.transform(feature_list)
    # ===========================  对分好类的所有patch进行分辨率提升 =======
    result = []

    a = time.time()
    print "开始构造KD树\n"

    # 将特征进行合并
    classified_feature = np.asarray(np.concatenate(classified_feature, axis=0))  # 特征的集合
    class_tag = np.asarray(np.concatenate(class_tag, axis=0)) #训练数据的集合

    orig_l_tree = KDTree(classified_feature, leaf_size=len(classified_feature) / 100)

    b = time.time()
    print "KD树构造完成：耗时%f秒\n" % (b - a)
    a = b

    # X除均方差
    mes = (np.sum(feature_list ** 2, axis=1)) ** 0.5 + eps
    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_knn = 1.0*feature_list / mes_list
    dist4, ind4 = orig_l_tree.query(feature_knn[:], k=16)  #16近邻885秒； 4近邻500秒

    b = time.time()
    print "搜索完成：耗时%f秒\n" % (b - a)
    a = b

    for ind in range(len(feature_list)):
        most = np.argsort(dist4[ind])
        lost = class_tag[ind4[ind]]
        patch_class = mode(lost[most].tolist())

        regressor = regression_list[np.int(patch_class)]

        res = np.dot(regressor, feature_list[ind])
        result.append(res)

    b = time.time()
    print "计算完成：耗时%f秒\n" % (b - a)
    a = b

    k = open("tmp_result_2times.pickle", 'wb')
    cPickle.dump(result, k, 1)
    k.close()
    return result


def get_super_patch_regression_without_search(feature_list, patch_list=None, tag=1, lim_list=[], mim_list=[]):
    """
    得到得到高分辨率的patch
    patch_list:需要进行超分辨率的patch
    class_list:该patch所在类别
    :return:
    """
    eps = np.finfo(float).eps

    # =========================== 加载回归模型以及分类好的数据 ===================
    regression_file = open('./tmp_file/%d_regression_ridge.pickle' % tag, 'rb')
    regression_list = cPickle.load(regression_file)
    regression_file.close()

    # ===========================  读取模型：pca以及kmeans ===============
    model_file = open('./tmp_file/%d_kmeans_pca_model.pickle' % 1, 'rb')
    (k_means, pca) = cPickle.load(model_file)
    model_file.close()

    # ===========================  进行测试patch分类 ====================
    class_list, feature_list = get_patch_class(feature_list, pca, k_means)
    feature_list = pca.transform(feature_list)
    # ===========================  对分好类的所有patch进行分辨率提升 =======
    result = []

    # X除均方差
    mes = (np.sum(feature_list ** 2, axis=1)) ** 0.5 + eps
    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_list = 1.0*feature_list / mes_list

    a = time.time()
    print "计算开始\n"

    for ind in range(len(feature_list)):
        result_tmp = np.dot(regression_list[0], feature_list[ind]).reshape((9, 9)) + mim_list[ind].reshape((9, 9))
        error = psnr(result_tmp.reshape((9, 9)), imresize(lim_list[ind].reshape((3, 3)), 3.0))

        for i in range(1, len(regression_list)):
            tmp = np.dot(regression_list[i], feature_list[ind]).reshape((9, 9)) + mim_list[ind].reshape((9, 9))
            e = psnr(tmp, imresize(lim_list[ind].reshape((3, 3)), 3.0))
            if e < error:
                error = e
                result_tmp = tmp

        # 从result_tmp中找到最接近lim_list的结果 放入result中
        result.append(result_tmp)

    b = time.time()
    print "计算完成：耗时%f秒\n" % (b - a)
    a = b

    k = open("tmp_result_without_search.pickle", 'wb')
    cPickle.dump(result, k, 1)
    k.close()
    return result


def get_super_patch_regression_with_superpixel(feature_list, patch_list=None, pixel_cls=None, tag=1):
    reg_dict = dict()
    eps = np.finfo(float).eps
    # =========================== 加载回归模型以及分类好的数据 ===================
    classified_file = open('./tmp_file/%d_class_with_regression.pickle' % tag, 'rb')
    classified_feature, classified_patch, classified_error, class_tag = cPickle.load(classified_file)
    classified_file.close()

    regression_file = open('./tmp_file/%d_regression_ridge.pickle' % tag, 'rb')
    regression_list = cPickle.load(regression_file)
    regression_file.close()

    # ===========================  读取模型：pca以及kmeans ===============
    model_file = open('./tmp_file/%d_kmeans_pca_model.pickle' % 1, 'rb')
    (k_means, pca) = cPickle.load(model_file)
    model_file.close()

    # ===========================  进行测试patch分类 ====================
    class_list, feature_list = get_patch_class(feature_list, pca, k_means)
    feature_list = pca.transform(feature_list)
    # ===========================  对分好类的所有patch进行分辨率提升 =======
    result = []

    a = time.time()
    print "开始构造KD树\n"

    # 将特征进行合并
    classified_feature = np.asarray(np.concatenate(classified_feature, axis=0))  # 特征的集合
    class_tag = np.asarray(np.concatenate(class_tag, axis=0)) #训练数据的集合

    orig_l_tree = KDTree(classified_feature, leaf_size=len(classified_feature) / 100)

    b = time.time()
    print "KD树构造完成：耗时%f秒\n" % (b - a)
    a = b

    # X除均方差
    mes = (np.sum(feature_list ** 2, axis=1)) ** 0.5 + eps
    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_knn = 1.0*feature_list / mes_list
    # dist4, ind4 = orig_l_tree.query(feature_knn[:], k=4)  #16近邻885秒； 4近邻500秒

    b = time.time()
    print "搜索完成：耗时%f秒\n" % (b - a)
    a = b

    for ind in range(len(feature_list)):
        if not reg_dict.has_key(pixel_cls[ind]):
            dist4, ind4 = orig_l_tree.query(feature_knn[ind], k=16)

            most = np.argsort(dist4[0])
            lost = class_tag[ind4[0]]
            patch_class = mode(lost[most].tolist())

            reg_dict[pixel_cls[ind]] = np.int(patch_class)

        regressor = regression_list[reg_dict[pixel_cls[ind]]]
        res = np.dot(regressor, feature_list[ind])

        result.append(res)

    b = time.time()
    print "计算完成：耗时%f秒\n" % (b - a)

    k = open("tmp_result.pickle", 'wb')
    cPickle.dump(result, k, 1)
    k.close()
    return result


def combine_image_deal_with_overlap(r_patch_list, pos_list, mim, mim_list, pl=9, overlap=6):
    """
    处理直接求平均值的问题
    :param patch_list:
    :param image:
    :return:
    """
    print r_patch_list[0].dtype
    # =============================== 使用回归的方法进行计算时的版本 =============================
    img = mim.copy() * 1.0
    mask = mim.copy() * 0.0+1.0000   # 记录叠加次数
    patch_r = mim.copy() * 0 - 1
    # point_r = [[[] for i in range(mim.shape[0])] for j in range(mim.shape[1])]  # 存放像素列表
    point_r = []  # 存放像素列表
    point_r_pos = []
    # patch_c = [[[] for i in range(mim.shape[0])] for j in range(mim.shape[1])]  # 存放patch列表
    patch_c = []  # 存放patch列表
    for ind, pos in enumerate(pos_list):
        # 使用搜索：
        img[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] = r_patch_list[ind].reshape((pl, pl)) + img[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] + mim_list[ind].reshape((pl, pl))

        patch_r[pos[0]][pos[1]] = ind

        mask[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] = mask[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] + 1.0

    for j in range(0, mim.shape[0]-pl * 2):
        for i in range(0, mim.shape[1]-pl * 2):
            point_r.append([])
            point_r_pos.append((j+8, i+8))
            # 通过这个循环计算出了所有重复点上的像素值
            for m in range(pl):
                for n in range(pl):
                    # 有的位置可能没有patch
                    if patch_r[j+m][i+n] < -0.1:
                        continue
                    point_r[-1].append(r_patch_list[np.int(patch_r[j + m][i + n])].reshape((pl, pl))[pl-m-1][pl-n-1])

    point_r = np.asarray([np.asarray(i) for i in point_r])

    # ============================ 在这里进行计算 从9个值当中选择最合适的一个====================
    mean = list(np.mean(point_r, axis=1))
    # print point_r
    std_my = point_r - np.repeat([np.mean(point_r, axis=1)], 9, axis=0).T
    max_ = list(np.max(point_r, axis=1))
    min_ = list(np.min(point_r, axis=1))

    point_sort = list(np.std(point_r, axis=1))
    #plt.hist(point_sort, bins=100)
    #plt.show()

    result_patchmean = img / mask
    gap = img * 0   # 单像素平均和patch平均的误差，为什么会有误差？ 最后结果的psnr值相差0.01 可以忽略
    # ======================================================================================
    for index, c in enumerate(point_sort):
        if c > 1.3:
            val = max(min_[index], mean[index] * 1.66)
            val = min(max_[index], val)
            img[point_r_pos[index][0]][point_r_pos[index][1]] = val + mim[point_r_pos[index][0]][point_r_pos[index][1]]
            #img[point_r_pos[index][0]][point_r_pos[index][1]] = 0.01
            mask[point_r_pos[index][0]][point_r_pos[index][1]] = 1.0
        if c > 1.8:
            gap[point_r_pos[index][0]][point_r_pos[index][1]] = c

    result_points = img / mask
    return result_patchmean, result_points, gap


def combine_image_from_patch(r_patch_list, pos_list, mim, mim_list, pl=9):
    """
    image是bicubic还原过以后的图像
    patch_list时通过稀疏编码计算出的需要累加的初始值
    :param patch_list:
    :param image:
    :return:
    """

    # =============================== 使用回归的方法进行计算时的版本 =============================
    img = mim.copy() * 1.0
    mask = mim.copy() * 0.0+1.0000   # 记录叠加次数
    for ind, pos in enumerate(pos_list):
        # 使用搜索：
        img[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] = r_patch_list[ind].reshape((pl, pl)) + img[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] + mim_list[ind].reshape((pl, pl))

        # 不是用搜索
        #print r_patch_list[ind]
        # img[pos[0]:pos[0]+pl, pos[1]:pos[1]+9] = r_patch_list[ind].reshape((9, 9)) + img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] #+ mim_list[ind].reshape((9, 9))
        mask[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] = mask[pos[0]:pos[0]+pl, pos[1]:pos[1]+pl] + 1.0
        pass
    '''
    # =============================== 使用稀疏编码的方法进行计算时的版本 =============================
    img = rim.copy() * 0
    mask = rim.copy() * 0+0.00001   # 记录叠加次数
    for ind, pos in enumerate(pos_list):
        img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = r_patch_list[ind].reshape((9, 9)) + img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] + np.mean(mim_list[ind].reshape((9, 9)))
        mask[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = mask[pos[0]:pos[0]+9, pos[1]:pos[1]+9] + 1
    '''
    return img / mask


def combine_image_from_patch_with_deformed(r_patch_list, pos_list, rim, mim_list, lim_list, patch_list,image=None):
    # ======        这里声明两个用于变形的目标和起始状态  ================
    lim_patch_list = []  # 低分辨率图像的patch
    n_patch_list = []  # 被修复的高分辨率图像的patch
    image_patch = []  # 图像的原始pathch
    rim_patch_list = []

    # =============================== 使用稀疏编码的方法进行计算时的版本 =============================
    for ind, pos in enumerate(pos_list):
        # ==================在这里可以使用变形，变形的过程发生在组合好有的patch上==========================
        lim_patch_list.append(lim_list[ind].reshape((3, 3)))
        rim_patch_list.append(mim_list[ind].reshape((9, 9)))
        image_patch.append(patch_list[ind].reshape((9, 9)) + np.mean(mim_list[ind].reshape((9, 9))))
        n_patch_list.append(r_patch_list[ind].reshape((9, 9))+ np.mean(mim_list[ind].reshape((9, 9))))

    f = open("deformed_source.pickle", "wb")
    cPickle.dump((lim_patch_list, rim_patch_list, image_patch, n_patch_list), f, 1)
    f.close()

    # '''
    # =========================================== 基于patch的方式进行 ============================
    #deformed_list = deformed_patch_based(lim_patch_list, rim_patch_list, image_patch, n_patch_list)
    img = rim.copy() * 1.0
    mask = rim.copy() * 0+1.00000   # 记录叠加次数
    for ind, pos in enumerate(pos_list):
        #img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = deformed_list[ind].reshape((9, 9)) + img[pos[0]:pos[0]+9, pos[1]:pos[1]+9]
        img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = r_patch_list[ind].reshape((9, 9)) + img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] +mim_list[ind].reshape((9, 9))
        mask[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = mask[pos[0]:pos[0]+9, pos[1]:pos[1]+9] + 1
    img = img / mask
    #'''

    '''
    # ========================================== 基于图像的方式进行
    img = rim.copy() * 0
    mask = rim.copy() * 0+0.00001   # 记录叠加次数
    for ind, pos in enumerate(pos_list):
        #img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = deformed_list[ind].reshape((9, 9)) + img[pos[0]:pos[0]+9, pos[1]:pos[1]+9]
        img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = r_patch_list[ind].reshape((9, 9)) + np.mean(img[pos[0]:pos[0]+9, pos[1]:pos[1]+9] + mim_list[ind].reshape((9, 9)))
        mask[pos[0]:pos[0]+9, pos[1]:pos[1]+9] = mask[pos[0]:pos[0]+9, pos[1]:pos[1]+9] + 1
    img = img / mask

    #img = deformed_image_based(img, image)
    # '''
    return img


def deformed_patch_based(lim_patch_list, rim_patch_list, image_patch, patch_list):
    deformed_list = []
    #print len(image_patch)
    #print len(patch_list)
    #print np.mean((np.asarray(image_patch)-np.asarray(patch_list))**2)**0.5

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
        #print np.mean((patch_list[i]-image_patch[i])**2)**0.5
        #print np.mean((cp-image_patch[i])**2)**0.5

        print "变形完成：%0.2f %%" % (i*1.0/len(lim_patch_list)*100)
    print np.mean((np.asarray(deformed_list)-np.asarray(patch_list))**2)**0.5
    return deformed_list


def deformed_image_based(src, dst):

    fx = np.asarray([[1.0, -1.0]], dtype='float')
    fy = np.asarray([[1.0], [-1.0]], dtype='float')

    grad_x = convolve2d(src, fx, mode='same')*0.1
    grad_y = convolve2d(src, fy, mode='same')*0.1

    grad_x[:, 0] = grad_x[:, 0]*0+1
    grad_y[0, :] = grad_y[0, :]*0+1

    Def = Deformed.deformed_patch()

    cp, c = Def.deform(src, dst,grad_x,grad_y)

    print psnr(src[:-9],dst[:-9])
    print psnr(cp[:-9],dst[:-9])

    return cp


def image_sharper(result_mean, theta=0.4):
    """
    范围需要是[0,255]
    """
    g_kernel = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
    result_sharp = cv2.filter2D(result_mean, -1, g_kernel)
    result_sharp = result_mean - result_sharp * theta
    result_sharp[result_sharp > 255] = 255
    result_sharp[result_sharp < 0] = 0
    return result_sharp


def devide_into_patch(feature, him, mim, patch_size=9, over_lap_size=6):
    feature_list = []
    him_list = []
    pos_list = []
    mim_list = []
    patch_mean = []

    patch_size_m = patch_size
    patch_size_h = patch_size
    over_lap_m = over_lap_size  # 7  # 可以随意的定义重叠的大小
    over_lap_h = over_lap_size  # 7  # 可以随意的定义重叠的大小

    size_m = mim.shape
    size_h = him.shape

    # xgrid_l = np.ogrid[0:size_l[0] - patch_size_l: patch_size_l - over_lap_l]
    # ygrid_l = np.ogrid[0:size_l[1] - patch_size_l: patch_size_l - over_lap_l]
    xgrid_m = np.ogrid[0:size_m[0] - patch_size_m: patch_size_m - over_lap_m]
    ygrid_m = np.ogrid[0:size_m[1] - patch_size_m: patch_size_m - over_lap_m]
    xgrid_h = np.ogrid[0:size_h[0] - patch_size_h: patch_size_h - over_lap_h]
    ygrid_h = np.ogrid[0:size_h[1] - patch_size_h: patch_size_h - over_lap_h]
    m = patch_size_m * patch_size_m * 4
    h = patch_size_h * patch_size_h
    # ========================     提取patch      ============================
    '''
        for x_l, x_m, x_h in zip(xgrid_l, xgrid_m, xgrid_h):
            for y_l, y_m, y_h in zip(ygrid_l, ygrid_m, ygrid_h):
        '''
    for x_m, x_h in zip(xgrid_m, xgrid_h):
        for y_m, y_h in zip(ygrid_m, ygrid_h):
            # 高清patch
            him_list.append(him[x_h:x_h + patch_size_h, y_h:y_h + patch_size_h].reshape((h,)))
            # 中清patch向量化
            mim_list.append(mim[x_h:x_h + patch_size_h, y_h:y_h + patch_size_h].reshape((h,)))
            # 特征patch
            feature_list.append(feature[:, x_m:x_m + patch_size_m, y_m:y_m + patch_size_m].reshape((m,)))
            # 高清patch的位置
            pos_list.append((x_h, y_h))  # 保存patch左上角的位置
            # 高清patch均值
            patch_mean.append(np.mean(mim[x_h:x_h + patch_size_h, y_h:y_h + patch_size_h].reshape((h,))))

    # ========================= 对patch进行前处理 ===========================
    feature_list, patch_list, mim_list = (np.asarray(feature_list, dtype=float),
                                          np.asarray(him_list, dtype=float),
                                          np.asarray(mim_list, dtype=float))
    return feature_list, him_list, mim_list, pos_list


def image_feature_generate(test_img):

    lim = imresize(test_img, 1 / 3.0, 'bicubic')
    mim = imresize(lim, 3.0, 'bicubic')
    lim = np.asarray(rgb2ycbcr(lim)[:, :, 0], dtype=float)
    him = np.asarray(rgb2ycbcr(test_img)[:, :, 0], dtype=float)
    mim = np.asarray(rgb2ycbcr(mim)[:, :, 0], dtype=float)
    feature = np.zeros((4, mim.shape[0], mim.shape[1]))
    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')
    feature[0, :, :] = convolve2d(mim, f1, mode='same')
    feature[1, :, :] = convolve2d(mim, f2, mode='same')
    feature[2, :, :] = convolve2d(mim, f3, mode='same')
    feature[3, :, :] = convolve2d(mim, f4, mode='same')
    return feature, him, lim, mim


def heavy_tailed_feature(img, color='red', label="result by patch mean"):
    # 计算梯度直方图
    h1, b1 = heavy_feature(img)
    plt.plot(b1, h1, color=color, label=label)


def show_result(him, r_gap, result, result_mean, result_sharp):
    mask = result.copy()
    mask[np.abs(him - result) > 10] = 0
    mask[np.abs(him - result) < 10] = 255
    plt.subplot(231)
    plt.imshow(him, interpolation="None", cmap=cm.gray)
    plt.subplot(232)
    plt.imshow(np.abs(him - result), interpolation="None", cmap=cm.gray)
    plt.subplot(233)
    plt.imshow(r_gap, interpolation="None", cmap=cm.gray)
    plt.subplot(234)
    plt.imshow(result_mean, interpolation="None", cmap=cm.gray)
    plt.subplot(235)
    plt.imshow(result, interpolation="None", cmap=cm.gray)
    plt.subplot(236)
    plt.imshow(result_sharp, interpolation="None", cmap=cm.gray)
    plt.show()


def compare_lim(lim, result_mean):
    lim_result = imresize(result_mean, 1 / 3.0, 'bicubic')
    plt.subplot(131)
    plt.imshow(lim_result, interpolation="None", cmap=cm.gray)
    plt.subplot(132)
    plt.imshow(lim, interpolation="None", cmap=cm.gray)
    plt.subplot(133)
    plt.imshow(np.abs(lim_result - lim), vmin=0, vmax=255, interpolation="None", cmap=cm.gray)
    plt.show()



"""
34.18313
"""