# coding:utf-8
import numpy as np
import cv2
import time
import random
import cPickle

import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

def distanse(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0],2) + np.power(p1[1] - p2[1],2))

def patch_dist(p1, p2):
    return np.sqrt(np.mean(np.power(p1 - p2,2)))

class RotationTrans(object):
    """输入patch_list,返回旋转的角度，需要能够锦进行正反变化"""
    pass
    def rotation(self, patch_list, patch_size=(21, 21), patch_center=(10,10)):
        degrees = []
        """
        degrees是通过计算质心的位置，让其位于最下方
        """
        mat_i = np.zeros(patch_size)
        mat_j = np.zeros(patch_size)
        for i in range(patch_size[0]):
            for j in range(patch_size[1]):
                mat_i[i][j] = i + 1
                mat_j[i][j] = j + 1

        for patch in patch_list:
            p = patch*self.inner_mask - np.min(patch*self.inner_mask)
            m = 1.0*np.sum(p)
            y = 1.0*np.sum(mat_i * p) / m - patch_center[0] - 1
            x = 1.0*np.sum(mat_j * p) / m - patch_center[1] - 1

            theta = np.arctan(-x*1.0/y)  * 180 / 3.14 # 这里计算的是逆时针的角度，而旋转是顺时针的旋转
            degrees.append(theta)
        return degrees

    def set_param(self, patch_size, patch_depth=1):
        """
        patch_size 必须为奇数
        :param patch_size:
        :param patch_depth:
        :return:
        """
        self.patch_size = patch_size
        self.patch_depth = patch_depth
        mask = np.zeros((patch_size, patch_size))
        inner_mask = np.zeros((patch_size, patch_size))
        min_mask = np.zeros((patch_size, patch_size))
        center = ((patch_size + 1) / 2, (patch_size + 1) / 2)
        self.center = center
        r = patch_size / 2

        gaussianKernel = cv2.getGaussianKernel(self.patch_size, 3)
        self.gaussianKernel2D = gaussianKernel * gaussianKernel.T

        for x in range(patch_size):
            for y in range(patch_size):
                if distanse((x + 1, y + 1), center) <= r:
                    mask[x, y] = 1.0
                if distanse((x + 1, y + 1), center) <= r - 2:
                    inner_mask[x, y] = 1.0
                if distanse((x + 1, y + 1), center) <= r - 4:
                    min_mask[x, y] = 1.0

        self.mask = mask
        self.min_mask = min_mask
        # 比正常的mask小一圈，为了消除在旋转当中产生的一些边界像素的误差
        self.inner_mask = inner_mask

    def get_rotation_version(self, patch, degrees, mask=None):
        if mask is None:
            mask = self.inner_mask
        r_patch_list = []
        for d in degrees:
            M = cv2.getRotationMatrix2D(self.center, d, 1)
            dst = cv2.warpAffine(patch, M, (self.patch_size, self.patch_size))
            r_patch_list.append(dst * mask)
        return r_patch_list

    def rotate_list(self, patch_list, degrees):
        return_list = []
        for i, d in enumerate(degrees):
            M = cv2.getRotationMatrix2D(self.center, d, 1)
            return_list.append(
                cv2.warpAffine(patch_list[i], M, (self.patch_size, self.patch_size)) * self.inner_mask)
        return return_list

    def get_degre_diff(self, patch_src, patch_dst, degree_step=1):
        """
        将patch_src向图像patch_dst对齐，同时返回patch_src旋转了得角度,
        最终决定具体要旋转多少度，一个是要查看旋转过去，在旋转回来的误差loss，另一个是要看旋转过去和目标的距离new_dist
        :param patch_src:
        :param patch_dst:
        :param degree_step: 两个角度之间的间隔
        :return:
        """

        patch_dst_degree_vector = np.reshape(patch_dst * self.min_mask * self.gaussianKernel2D,
                                             (1, self.patch_size * self.patch_size))
        # 得到两个patch的最小角度差，这个可不可以用神经网络进行训练？ 输入时两个图像，输出是两个图像时间的角度，是一个双通道的图像
        degrees = range(0, 360, degree_step)
        patch_src_degrees_list = self.get_rotation_version(patch_src, degrees,
                                                           self.min_mask * self.gaussianKernel2D)
        # for i in patch_src_degrees_list:
        #     plt.imshow(i)
        #     plt.show()

        patch_src_degrees_mat = np.reshape(np.array(patch_src_degrees_list),
                                           (-1, self.patch_size * self.patch_size))
        d = np.argmin(np.sum(np.power((patch_src_degrees_mat - patch_dst_degree_vector), 2), axis=1))

        patch_src_rotated = self.get_rotation_version(patch_src, [d])[0]
        patch_src_back = self.get_rotation_version(patch_src_rotated, [-d])[0]
        old_dist = patch_dist(patch_dst * self.min_mask, patch_src * self.min_mask)
        new_dist = patch_dist(patch_dst * self.min_mask, patch_src_rotated * self.min_mask)
        loss = patch_dist(patch_src * self.min_mask, patch_src_back * self.min_mask)

        # print("%.2f=>>%.2f     %.2f" % (old_dist,new_dist,loss))
        # plt.subplot(131)
        # plt.imshow(patch_src* self.min_mask)
        # plt.subplot(132)
        # plt.imshow(patch_src_back* self.min_mask)
        # plt.subplot(133)
        # plt.imshow(patch_dst * self.min_mask)
        # plt.show()

        return d, new_dist, loss

    def patch_partition(self, patch_list, n_clusters=30):
        """
        输入path列表，返回聚类中心，和聚类标签
        :param patch_list:
        :param n_clusters:
        :return:
        """
        patch_data = np.array(patch_list)
        patch_data = np.reshape(patch_data, (-1, self.patch_size * self.patch_size * self.patch_depth))
        k_means = KMeans(n_clusters=n_clusters)
        t0 = time.time()
        k_means.fit(patch_data)
        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(patch_data, k_means_cluster_centers)
        t_batch = time.time() - t0
        print ("%d个patch进行Kmean聚类,分成%d类，耗时%d秒" % (patch_data.shape[0], n_clusters, t_batch))
        return np.reshape(np.array(k_means_cluster_centers), (-1, 21, 21)), k_means_labels

if __name__ == "__main__":
    """
    TODO:
    1.【finished】需要检查mask是否合理的使用了，需要对所有的patch使用mask转换成圆形的patch
    2.目前的数据集和都是十分稀少的，数量在1W左右的patch计算,但是实际计算的patch数据量在100W需要进行优化，或者层次计算
    3.【finished】需要对patch进行正确的归一化处理，例如减去均值，合理的距离计算等操作。
    4.【finished】在计算距离的时候使用的高斯核，中间的权重大，两边的权重小

    """
    input_tag = "291"
    output_tag = "291_cnn_Y_channel_21.pic"
    res_path = 'E:/mySuperResolution/dataset/%s/%s' % (input_tag, output_tag)

    print res_path
    f = open(res_path, 'rb')
    t = cPickle.load(f)
    f.close()

    r = RotationTrans()
    r.set_param(21, 1)
    n_clusters = 30
    training_data = t
    seg_num = 5000

    train_input = training_data[0][0:120000]
    train_output = training_data[1][0:120000]
    data_len = len(train_input)
    D = np.zeros((data_len))
    back_input = r.rotate_list(train_input, D)

    for k in range(0, data_len ,seg_num):
        print k
        input = np.array(train_input[k: min(k+ seg_num,data_len)])
        output = np.array(train_output[k: min(k + seg_num,data_len)])

        # 根据质心旋转
        D = np.array(r.rotation(input))
        dst_input = r.rotate_list(input, D)

        for itr in range(60):
            tmp_D = np.zeros(input.shape[0])
            tmp_degress = np.zeros((n_clusters))
            t = time.time()
            # 这个中心是旋转以后的中心
            center, labels = r.patch_partition(dst_input, n_clusters=n_clusters)
            dis = np.ones((n_clusters)) * 1000
            dis[0] = 0
            for i in range(1, n_clusters):
                for j in range(i):
                    if i == j:
                        continue
                    d, error, loss = r.get_degre_diff(center[i], center[j])
                    if dis[i] > error + 0.1 * loss:
                        dis[i] = error + 0.1 * loss
                        tmp_degress[i] = d + tmp_degress[j]
            # 可能大于360
            # print (tmp_degress)
            for i in range(D.shape[0]):
                tmp_D[i] = tmp_degress[labels[i]]  # 这里的D需要通过我们计算的label确定每一个patch旋转多少度

            D = D + tmp_D
            D = D % 360
            print("第%d次迭代耗时%.2f秒" % (itr, time.time() - t))
            t = time.time()

            dst_input = r.rotate_list(input, D)

        train_input[k: min(k + seg_num, data_len)] = r.rotate_list(input, D)
        train_output[k: min(k + seg_num, data_len)]= r.rotate_list(output,D)
        # 这里的D就是每一个patch需要旋转的角度
        print(D)

    # file = open("rotated_patch.cp", "wb")
    # cPickle.dump((src_input, dst_input, D), file)
    # file.close()

    rotated_tag = "291_cnn_Y_channel_21_rotated.pic"
    file = open('E:/mySuperResolution/dataset/%s/%s-v1' % (input_tag, rotated_tag),'wb')
    cPickle.dump((train_input, train_output), file)
    file.close()

    # 显示数据
    h = 1000
    w = 1000
    ind = 0
    conv = np.zeros((h, w))
    for i in range(1, h - 25, 25):
        for j in range(1, w - 25, 25):
            if ind > data_len - 1 :
                break
            conv[i:i + 21, j:j + 21] = back_input[ind]
            ind = ind + 1

    ind = 0
    conv_rotated = np.zeros((h, w))
    for i in range(1, h - 25, 25):
        for j in range(1, w - 25, 25):
            if ind > data_len - 1 :
                break
            conv_rotated[i:i + 21, j:j + 21] = train_input[ind]
            ind = ind + 1

    plt.subplot(121)
    plt.imshow(conv)
    plt.subplot(122)
    plt.imshow(conv_rotated)
    plt.show()

