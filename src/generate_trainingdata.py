# coding:utf-8
import numpy as np
import skimage.io as io
import cv2
import skimage.filters as ft
import os
from scipy.misc import imresize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.morphology import disk
from toolgate.colormanage import rgb2ycbcr, ycbcr2rgb
import cPickle
__author__ = 'liangz14'


def get_train_set_by_scale(img_lib, input_size, output_size, over_lap=1):
    # 抽取字典~这个字典应该是HR-LR堆叠起来的结果
    scale = output_size / input_size  # 2 或 1.5

    # 训练数据列表
    feature_lib = []
    target_lib = []
    raw_lib = []

    # 计算4个特征的卷积核,可以用4个方向的滤波作为特征
    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')

    # 保存图像列表
    feature_img_list = []
    him_img_list = []  # 高ingpatch
    target_img_list = []

    # 每张图片计算八组特征
    for img in img_lib:
        s = img.shape
        # 要处理的原始图像可以被6整除，这样就可以缩小1.5或者2倍
        image = img[0:s[0] - s[0] % 6, 0:s[1] - s[1] % 6, :]

        lim = imresize(image, 1.0/scale, interp='bicubic')
        # 缩小放大后的图像，用于提取训练特征
        mim = imresize(lim, scale, interp='bicubic')

        # 提取y通道的方式进行计算 y[16 235] cbcr [16 240]
        him = np.asarray(rgb2ycbcr(image)[:, :, 0], dtype=float)
        mim = np.asarray(rgb2ycbcr(mim)[:, :, 0], dtype=float)
        target_img = him - mim
        '''
        plt.subplot(121)
        plt.imshow(mim)
        plt.subplot(122)
        plt.imshow(him)
        plt.show()
        '''
        feature_img = np.zeros((4, mim.shape[0], mim.shape[1]))

        feature_img[0, :, :] = convolve2d(mim, f1, mode='same')
        feature_img[1, :, :] = convolve2d(mim, f2, mode='same')
        feature_img[2, :, :] = convolve2d(mim, f3, mode='same')
        feature_img[3, :, :] = convolve2d(mim, f4, mode='same')

        feature_img_list.append(feature_img)
        target_img_list.append(target_img)
        him_img_list.append(him)

    for i in zip(feature_img_list, target_img_list, him_img_list):
        size_m = i[0].shape[1:]
        size_h = i[1].shape

        xgrid_m = np.ogrid[0:size_m[0]-output_size: output_size - over_lap]
        ygrid_m = np.ogrid[0:size_m[1]-output_size: output_size - over_lap]
        xgrid_h = np.ogrid[0:size_h[0]-output_size: output_size - over_lap]
        ygrid_h = np.ogrid[0:size_h[1]-output_size: output_size - over_lap]

        m = output_size * output_size * 4
        h = output_size * output_size

        for x_m, x_h in zip(xgrid_m, xgrid_h):
            for y_m, y_h in zip(ygrid_m, ygrid_h):
                target_lib.append(i[1][x_h:x_h+output_size, y_h:y_h+output_size].reshape((h,)))
                feature_lib.append(i[0][:, x_m:x_m+output_size, y_m:y_m+output_size].reshape((m,)))
                raw_lib.append(i[2][x_h:x_h+output_size, y_h:y_h+output_size].reshape((h,)))

    return target_lib, feature_lib, raw_lib


def get_train_set(L_channel_lis, scale=3.0, feat_scale=3.0, patch_size_l=3):
    # 抽取字典~这个字典应该是HR-LR堆叠起来的结果
    patch_size_m = feat_scale * patch_size_l  # 9
    patch_size_h = scale * patch_size_l  # 9

    over_lap_l = 1  # 1
    over_lap_m = feat_scale * over_lap_l  # 3
    over_lap_h = scale * over_lap_l  # 3

    feature_lib = []
    target_lib = []
    raw_lib = []

    # 计算4个特征的卷积核,可以用4个方向的滤波作为特征
    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')

    # 保存图像列表
    feature_img_list = []
    him_img_list = []  # 高ingpatch
    target_img_list = []

    # 每张图片计算八组特征
    for img in L_channel_lis:
        s = img.shape
        # 要处理的原始图像
        image = img[0:s[0] - s[0] % 3, 0:s[1] - s[1] % 3, :]

        lim = imresize(image, 1.0/scale, interp='bicubic')
        # 缩小放大后的图像，用于提取训练特征
        mim = imresize(lim, scale, interp='bicubic')

        # 提取y通道的方式进行计算 y[16 235] cbcr [16 240]
        him = np.asarray(rgb2ycbcr(image)[:, :, 0], dtype=float)
        mim = np.asarray(rgb2ycbcr(mim)[:, :, 0], dtype=float)
        target_img = him - mim

        feature_img = np.zeros((4, mim.shape[0], mim.shape[1]))

        feature_img[0, :, :] = convolve2d(mim, f1, mode='same')
        feature_img[1, :, :] = convolve2d(mim, f2, mode='same')
        feature_img[2, :, :] = convolve2d(mim, f3, mode='same')
        feature_img[3, :, :] = convolve2d(mim, f4, mode='same')

        # 改进特征1 :进行开方
        #featurn_sign = np.sign(feature_img)  # 提取符号
        #feature_img = np.sqrt(np.abs(feature_img)) * featurn_sign

        # 改进特征2 :进行平方
        #featurn_sign = np.sign(feature_img)  # 提取符号
        #feature_img = np.abs(feature_img)**2 * featurn_sign

        #
        feature_img_list.append(feature_img)
        # target_img_list.append(image_pre_filter(target_patch))
        target_img_list.append(target_img)
        him_img_list.append(him)

    for i in zip(feature_img_list, target_img_list, him_img_list):
        size_m = i[0].shape[1:]
        size_h = i[1].shape

        xgrid_m = np.ogrid[0:size_m[0]-patch_size_m: patch_size_m - over_lap_m]
        ygrid_m = np.ogrid[0:size_m[1]-patch_size_m: patch_size_m - over_lap_m]
        xgrid_h = np.ogrid[0:size_h[0]-patch_size_h: patch_size_h - over_lap_h]
        ygrid_h = np.ogrid[0:size_h[1]-patch_size_h: patch_size_h - over_lap_h]

        m = patch_size_m * patch_size_m * 4
        h = patch_size_h * patch_size_h

        for x_m, x_h in zip(xgrid_m, xgrid_h):
            for y_m, y_h in zip(ygrid_m, ygrid_h):
                target_lib.append(i[1][x_h:x_h+patch_size_h, y_h:y_h+patch_size_h].reshape((h,)))
                feature_lib.append(i[0][:, x_m:x_m+patch_size_m, y_m:y_m+patch_size_m].reshape((m,)))
                raw_lib.append(i[2][x_h:x_h+patch_size_h, y_h:y_h+patch_size_h].reshape((h,)))

    return target_lib, feature_lib, raw_lib


def get_train_set_for_cnn(mat_lib, patch_size=21, over_lap=5, scale=3.0) :
    """
    这里主要是为卷积神经网络准备数据，
    :param mat_lib:  输入的可以是彩色图片，也可以是L通道的图片, 先处理L通道
    :param patch_size: 输出的结果是根据这个尺寸定义的patch，主要用于神经网络的讯息 在vdsr当中大小是41 41 我使用21 21
    :param over_lap: patch采样的覆盖
    :return: 返回的是patch对，大小一样只不过一个是清楚的一是模糊的，vdsr当中使用的多个模糊程度对应一个清楚。这个可以考虑进来
    """
    cnn_output = []
    cnn_input = []
    i = 0
    all = len(mat_lib)
    for Y_H, Y_L in mat_lib:
        Y_H = Y_H[0: Y_H.shape[0] - Y_H.shape[0] % 3, 0: Y_H.shape[1] - Y_H.shape[1] % 3]
        Y_L = Y_L[0: Y_L.shape[0] - Y_L.shape[0] % 3, 0: Y_L.shape[1] - Y_L.shape[1] % 3]

        out = Y_H - Y_L

        size = Y_H.shape

        xgrid = np.ogrid[0:size[0] - patch_size: patch_size - over_lap]
        ygrid = np.ogrid[0:size[1] - patch_size: patch_size - over_lap]

        for x in xgrid:
            for y in ygrid:
                LR_patch = Y_L[x:x+patch_size, y:y+patch_size] - np.mean(Y_L[x:x+patch_size, y:y+patch_size])
                HR_patch = out[x:x+patch_size, y:y+patch_size]

                cnn_input.append(LR_patch)
                cnn_output.append(HR_patch)
        i = i + 1
        print("已处理%.2f" % (i * 1.0 / all * 100.0))
    return np.array(cnn_input), np.array(cnn_output)


def image_pre_filter(img):
    """
    对输入的单通道图像进行预处理：
    包括中值滤波，锐化，降噪等操作，办证patch的效果
    :param img:
    :return:
    """

    result = ft.median(img / 255.0, disk(2))
    gs = ft.gaussian_filter(result, 1.)*255.0
    result += (result - gs) * 1.0
    # result[result>255] = 255
    # result[result<0] = 0

    # plt.subplot(121)
    # plt.imshow(img, interpolation="none", cmap=cm.gray)
    # plt.subplot(122)
    # plt.imshow(result, interpolation="none", cmap=cm.gray)
    # plt.show()
    return result


def read_img_train_Y_channel(cur_dir, down_time=20, scale=0.97):
    """
    读取目录下的图片:提取全部图片，并提取成单通道图像，并归一化数值[0,1]
    :param cur_dir:
    :return:
    """
    img_file_list = os.listdir(cur_dir)  # 读取目录下全部图片文件名

    img_lib = []
    for file_name in img_file_list:
        if file_name[-3:] != "jpg" and file_name[-3:] != "bmp":
            continue
        full_file_name = os.path.join(cur_dir, file_name)
        image = cv2.imread(full_file_name)  # 读取一张图片

        image = image[0:image.shape[0] - image.shape[0] % 3, 0:image.shape[1] - image.shape[1] % 3,:]
        image_low = imresize(image, 1/3.0, interp='bicubic')
        image_r = imresize(image_low, 3.0, interp='bicubic')

        image_h = np.array(image/255.0, dtype=np.float32)
        image_l = np.array(image_r / 255.0, dtype=np.float32)

        Y_H = cv2.cvtColor(image_h, cv2.COLOR_BGR2YCR_CB, None)[:, :, 0]
        Y_L = cv2.cvtColor(image_l, cv2.COLOR_BGR2YCR_CB, None)[:, :, 0]

        fac = 1.0
        # 多次下采样作为样本：
        for i in range(down_time):
            if fac < 0.6 :
                break
            _Y_H = cv2.resize(Y_H,(0, 0), None, fac, fac)
            _Y_L = cv2.resize(Y_L, (0, 0), None, fac, fac)
            fac = fac * scale
            img_lib.append((np.array(_Y_H, dtype = np.float32), np.array(_Y_L, dtype = np.float32)))
    print("图像字典完毕，共%d对图片" % len(img_lib))
    return img_lib


def read_img_train_RGB(cur_dir, down_time=10, scale=0.92):
    """
    读取目录下的图片:提取全部图片，并提取成单通道图像，并归一化数值[0,1]
    :param cur_dir:
    :return:
    """
    img_file_list = os.listdir(cur_dir)  # 读取目录下全部图片文件名

    img_lib = []
    for file_name in img_file_list:
        if file_name[-3:] != "jpg" and file_name[-3:] != "bmp":
            continue
        full_file_name = os.path.join(cur_dir, file_name)
        image = cv2.imread(full_file_name)  # 读取一张图片
        fac = scale
        # 多次下采样作为样本：
        for i in range(down_time):
            _image = cv2.resize(image,(0, 0), None, fac, fac)
            fac = fac * scale
            if np.min(image.shape[0:2]) > np.min(_image.shape[0:2]) * 2:
                break
            img_lib.append(np.array(_image, dtype = np.float32))
    return img_lib

def main_generate(input_tag="B100",output_tag="training", tr_num=800000):
    image_data_path = "E:/mySuperResolution/dataset/%s" % input_tag
    res_path = 'E:/mySuperResolution/dataset/%s/%s' % (input_tag, output_tag)
    res_raw_patch_path = 'E:/mySuperResolution/dataset/%s/%s' % (input_tag, output_tag)

    # 读取所有的图片, 已经经过图像颜色通道处理，提取出了L通道
    # L_channel_lis = read_img_train_L_channel(image_data_path)
    # 读取所有的图片, 未经过图像颜色通道处理
    img_lib = read_img_train_RGB(image_data_path)
    #
    # for im in img_lib:
    #     im = cv2.normalize(im,None, 0, 1, cv2.NORM_MINMAX)
    #     cv2.imshow("a", im)
    #     cv2.waitKey(0)
    # print len(L_channel_lis)

    # 得到训练的输入（梯度等特征），输出（可能经过的归一化等处理）和原始图像的patch
    # patch_lib, feature_lib, raw_lib = get_train_set_by_scale(img_lib, input_size=6.0, output_size=9.0, over_lap=3)
    # patch_lib, feature_lib, raw_lib = get_train_set_by_scale(img_lib, input_size=3.0, output_size=6.0, over_lap=2)
    patch_lib, feature_lib, raw_lib = get_train_set(img_lib)

    if len(patch_lib) > tr_num:
        s_factor = len(patch_lib)/tr_num + 1
        patch_lib = patch_lib[::s_factor]
        feature_lib = feature_lib[::s_factor]
        raw_lib = raw_lib[::s_factor]

    training_data = (patch_lib, feature_lib)

    with open(res_raw_patch_path, 'wb') as f:
        cPickle.dump(raw_lib, f, 1)

    with open(res_path, 'wb') as f:
        cPickle.dump(training_data, f, 1)

    print "训练数据已经保存！"
    ''''''
    print patch_lib[1].shape
    print feature_lib[1].shape
    print len(patch_lib)
    print len(feature_lib)

def main_generate_CNN(input_tag="B100",output_tag="B100_cnn_L_channel.pic", tr_num=500000):
    image_data_path = "E:/mySuperResolution/dataset/%s" % input_tag
    res_path = 'E:/mySuperResolution/dataset/%s/%s' % (input_tag, output_tag)

    # 读取所有的图片, 已经经过图像颜色通道处理，提取出了L通道
    Y_channel_lis = read_img_train_Y_channel(image_data_path)
    # 读取所有的图片, 未经过图像颜色通道处理
    # img_lib = read_img_train_RGB(image_data_path)
    #
    # for im in img_lib:
    #     im = cv2.normalize(im,None, 0, 1, cv2.NORM_MINMAX)
    #     cv2.imshow("a", im)
    #     cv2.waitKey(0)
    # print len(L_channel_lis)

    # 得到训练的输入（梯度等特征），输出（可能经过的归一化等处理）和原始图像的patch
    # patch_lib, feature_lib, raw_lib = get_train_set_by_scale(img_lib, input_size=6.0, output_size=9.0, over_lap=3)
    # patch_lib, feature_lib, raw_lib = get_train_set_by_scale(img_lib, input_size=3.0, output_size=6.0, over_lap=2)
    cnn_input, cnn_output = get_train_set_for_cnn(Y_channel_lis,patch_size=41,over_lap=0)
    print cnn_output.shape
    print cnn_input.shape
    if len(cnn_input) > tr_num:
        s_factor = len(cnn_input)/tr_num + 1
        cnn_input = cnn_input[::s_factor]
        cnn_output = cnn_output[::s_factor]

    training_data = (cnn_input, cnn_output)

    if os.path.exists(res_path):
        os.remove(res_path)
    f = open(res_path, 'wb')
    cPickle.dump(training_data, f,1)
    f.close()
    print "训练数据已经保存！"
    print cnn_input.shape
    print res_path

if __name__ == '__main__':
    # 修改这个结果就可以生成不同文件夹中的训练数据
    # main_generate( tr_num=800000)


    input_tag = "291"
    output_tag = "291_cnn_Y_channel_41.pic"
    main_generate_CNN(tr_num=500000, input_tag =input_tag,output_tag=output_tag)
    res_path = 'E:/mySuperResolution/dataset/%s/%s' % (input_tag, output_tag)

    print res_path
    f = open(res_path, 'rb')
    t = cPickle.load(f)
    f.close()
    print t[1].shape
