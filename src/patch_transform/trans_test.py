# coding:utf-8
from affine import *
from rotation import *
from movement import *
import matplotlib.pyplot as plt
import cPickle

if __name__ == "__main__":
    input_tag = "291"
    output_tag = "291_cnn_Y_channel.pic"
    res_path = 'E:/mySuperResolution/dataset/%s/%s' % (input_tag, output_tag)

    print res_path
    f = open(res_path, 'rb')
    t = cPickle.load(f)
    f.close()
    patch_list = t[0][0:10000]
    r = RotationTrans()
    rotated_patch_list, degree = r.rotation(patch_list)

    #显示数据
    h = 1000
    w = 1000
    ind = 0
    conv = np.zeros((h,w))
    for i in range(1,h - 25, 25):
        for j in range(1, w - 25, 25):
            conv[i:i + 21, j:j + 21] = patch_list[ind]- np.min( patch_list[ind])
            ind = ind + 1

    ind = 0
    conv_rotated = np.zeros((h,w))
    for i in range(1,h - 25, 25):
        for j in range(1, w - 25, 25):
            conv_rotated[i:i + 21, j:j + 21] = rotated_patch_list[ind]- np.min( rotated_patch_list[ind])
            ind = ind + 1

    plt.subplot(121)
    plt.imshow(conv)
    plt.subplot(122)
    plt.imshow(conv_rotated)
    plt.show()