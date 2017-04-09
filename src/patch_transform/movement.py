# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_patch_list(patch_list):
    h = 1000
    w = 1000

    ind = 0
    conv = np.zeros((h, w))
    for i in range(1, h - 25, 25):
        for j in range(1, w - 25, 25):
            if ind >= len(patch_list)-1:
                break
            conv[i:i + 21, j:j + 21] = patch_list[ind]
            ind = ind + 1

    plt.imshow(conv)



def get_patch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, None)
    patch_list = []
    size = gray.shape
    print size
    x_grid = np.ogrid[0:size[0] - 21:4]
    y_grid = np.ogrid[0:size[1] - 21:4]

    for i in x_grid:
        for j in y_grid:
            patch_list.append(gray[i: i + 21,j:j + 21])
    return patch_list

def get_1D_sample_grid(begin, end, center, point_num):
    p = []
    num = (point_num - 1) / 2
    rate1 = (center - begin) / num
    rate2 = (end - center) / num
    pos = begin
    for i in range(num + 1):
        p.append(pos)
        pos = pos + rate1
    for i in range(num):
        p.append(pos)
        pos = pos + rate2
    return np.array(p,dtype=np.float32)

def get_moved_patch(patch, patch_size=(21, 21), patch_center=(10,10), move_center=(10,10)):
    x_grid = get_1D_sample_grid(0, patch_size[0] - 1, move_center[0], patch_size[0])
    y_grid = get_1D_sample_grid(0, patch_size[1] - 1, move_center[1], patch_size[1])
    moved_patch = np.zeros_like(patch)
    for x, new_x in enumerate(x_grid):
        for y, new_y in enumerate(y_grid):
            moved_patch[x,y] = patch[min(patch_size[0] -1 ,int(new_x)),min(patch_size[1] -1 ,int(new_y))]
    return moved_patch, x_grid, y_grid

def get_m_center(patch, patch_size=(21, 21), patch_center=(10,10)):
    """
    回去质心位置
    :param patch:
    :return:
    """
    kernel = np.zeros((3,3),dtype=np.float32)
    kernel[1,1] = -4
    kernel[2, 1] = 1
    kernel[0, 1] = 1
    kernel[1, 2] = 1
    kernel[1, 0] = 1
    tmp_p = cv2.filter2D(patch, cv2.CV_8UC1, kernel)
    tmp_p = np.copy(tmp_p) * 1.0
    tmp_p = tmp_p - np.min(tmp_p)+ 1
    mat_i = np.zeros(patch_size,dtype=np.float32)
    mat_j = np.zeros(patch_size,dtype=np.float32)
    for i in range(patch_size[0]):
        for j in range(patch_size[1]):
            mat_i[i][j] = i + 1
            mat_j[i][j] = j + 1
    x_feature = np.sum(tmp_p * mat_i) / np.sum(tmp_p) - 1
    y_feature = np.sum(tmp_p * mat_j) / np.sum(tmp_p) - 1

    return x_feature,y_feature

def get_mfeature(img, patch_size=(21, 21), patch_center=(10,10)):
    kernel = np.zeros((3,3))
    kernel[1,1] = -4
    kernel[2, 1] = 1
    kernel[0, 1] = 1
    kernel[1, 2] = 1
    kernel[1, 0] = 1
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, None)
    img = cv2.filter2D(img, cv2.CV_8UC1, kernel)
    mat_i = np.zeros(patch_size)
    mat_j = np.zeros(patch_size)
    for i in range(patch_size[0]):
        for j in range(patch_size[1]):
            mat_i[i][j] = i + 1
            mat_j[i][j] = j + 1

    size = img.shape
    x_feature = np.zeros_like(img,dtype=np.float32)
    y_feature = np.zeros_like(img,dtype=np.float32)
    for i in range(size[0] - 21):
        for j in range(size[1] - 21):
            p = img[i:i+21, j:j+21]* 1.0
            x_feature[i,j] = np.sum(1.0/p * mat_i)/ np.sum(1.0/p) - patch_center[0] - 1
            y_feature[i,j] = np.sum(1.0/p * mat_j)/ np.sum(1.0/p) - patch_center[1] - 1
    x_feature = x_feature
    y_feature = y_feature
    a = x_feature * x_feature + y_feature * y_feature
    plt.subplot(141)
    plt.imshow(x_feature*1.0,cmap="gray")
    plt.subplot(142)
    plt.imshow(y_feature*1.0,cmap="gray")
    plt.subplot(143)
    plt.imshow(img,cmap="gray")
    plt.subplot(144)
    plt.imshow(a,cmap="gray")
    plt.show()



class MovementTrans(object):
    """输入patchlist 返回patch list和位移的多少，能够正反变化"""
    pass


if __name__ == "__main__":
    image = cv2.imread("E:\\mySuperResolution\\dataset\\B100\\148026.jpg")
    # get_mfeature(image)
    patch_list = get_patch(image)
    moved_patch_list = []
    xs = []
    ys = []
    mxs = []
    mys = []
    for index, patch in enumerate(patch_list):
        if index > 3000:
            break
        x, y = get_m_center(patch)

        ys.append(y - 10)
        xs.append(x - 10)
        moved_patch, a , b = get_moved_patch(patch,move_center=(x, y))
        moved_patch_list.append(moved_patch)

        mx, my = get_m_center(moved_patch)
        mys.append(my - 10)
        mxs.append(mx - 10)
    plt.subplot(121)
    show_patch_list(patch_list)
    plt.subplot(122)
    show_patch_list(moved_patch_list)
    plt.show()
    plt.subplot(221)
    plt.plot(range(len(xs)),xs,".")
    plt.subplot(222)
    plt.plot(range(len(ys)),ys,".")

    plt.subplot(223)
    plt.plot(range(len(mxs)),mxs,".")
    plt.subplot(224)
    plt.plot(range(len(mys)),mys,".")
    plt.show()
