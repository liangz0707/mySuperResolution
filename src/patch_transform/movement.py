# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_patch(img):
    h = 1000
    w = 1000
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, None)
    patch_list = []
    size = img.shape[1:3]
    x_grid = np.ogrid[0:size[0] - 21:4]
    y_grid = np.ogrid[0:size[1] - 21:4]

    ind = 0
    for i in x_grid:
        for j in y_grid:
            patch_list.append(gray[i: i + 21][j:j + 21])

    conv = np.zeros((h, w))
    for i in range(1, h - 25, 25):
        for j in range(1, w - 25, 25):
            if ind >= len(patch_list)-1:
                break
            conv[i:i + 21, j:j + 21] = patch_list[ind]
            ind = ind + 1

    plt.subplot(121)
    plt.imshow(conv)
    plt.subplot(122)
    plt.show()


class MovementTrans(object):
    """输入patchlist 返回patch list和位移的多少，能够正反变化"""
    pass


if __name__ == "__main__":
    image = cv2.imread("/Users/liangz14/workspace/Python/mySuperResolution/dataset/IMG_4041.png")
    image = cv2.resize(image, (500, 500))
    get_patch(image)

