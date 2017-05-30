# coding:utf-8
from sr_tools import *
__author__ = 'liangz14'


def super_image_mullayer(method="regression", input='result/baby_GT.bmp', output="result/baby_MY[gray]reg_3times.bmp"):
    """
    处理超分辨率的输入输出
    :return:
    """
    # ================================得到超分辨率的输入和输出=================================
    # =======================  **.jpg ---》》 feature and patch ====================
    raw_img = io.imread(input)
    # raw_img = raw_img[120:240, 80:200, :]
    shape = raw_img.shape
    # 首先需要降低三倍
    if len(shape) == 3:
        raw_img = raw_img[:shape[0]-shape[0] % 3, :shape[1]-shape[1] % 3, :]
    else:
        raw_img = raw_img[:shape[0]-shape[0] % 3, :shape[1]-shape[1] % 3]

    # 首先提升两倍

    lim = imresize(raw_img, 1/3.0, 'bicubic')
    mim = imresize(lim, 2.0, 'bicubic')

    lim = np.asarray(rgb2ycbcr(lim)[:, :, 0], dtype=float)
    mim = np.asarray(rgb2ycbcr(mim)[:, :, 0], dtype=float)
    him = np.asarray(rgb2ycbcr(raw_img)[:, :, 0], dtype=float)

    feature = np.zeros((4, mim.shape[0], mim.shape[1]))

    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')

    feature[0, :, :] = convolve2d(mim, f1, mode='same')
    feature[1, :, :] = convolve2d(mim, f2, mode='same')
    feature[2, :, :] = convolve2d(mim, f3, mode='same')
    feature[3, :, :] = convolve2d(mim, f4, mode='same')

    # mim_class = super_pixel_classify(test_img, cls_num=100)

    # ================================ 将patch 和feature 分解list=================================

    feature_list = []
    him_list = []
    pos_list = []
    patch_mean = []
    lim_list = []
    mim_list = []
    pos_list_lim = []
    mim_cls_list = []

    patch_size_m = 6
    patch_size_h = 6
    patch_size_l = 3
    over_lap_l = 1
    over_lap_m = 2
    over_lap_h = 2

    size_l = lim.shape
    size_m = mim.shape

    xgrid_l = np.ogrid[0:size_l[0] - patch_size_l: patch_size_l - over_lap_l]
    ygrid_l = np.ogrid[0:size_l[1] - patch_size_l: patch_size_l - over_lap_l]
    xgrid_m = np.ogrid[0:size_m[0] - patch_size_m: patch_size_m - over_lap_m]
    ygrid_m = np.ogrid[0:size_m[1] - patch_size_m: patch_size_m - over_lap_m]

    m = patch_size_m * patch_size_m * 4
    h = patch_size_h * patch_size_h
    l = patch_size_l * patch_size_l

    # ========================     提取patch      ============================
    for x_l, x_m in zip(xgrid_l, xgrid_m):
        for y_l, y_m in zip(ygrid_l, ygrid_m):
            # 中清patch向量化
            mim_list.append(mim[x_m:x_m+patch_size_m, y_m:y_m+patch_size_m].reshape((h,)))
            # 低清patch向量化
            lim_list.append(lim[x_l:x_l+patch_size_l, y_l:y_l+patch_size_l].reshape((l,)))
            # 特征patch
            feature_list.append(feature[:, x_m:x_m+patch_size_m, y_m:y_m+patch_size_m].reshape((m,)))
            # 高清patch的位置
            pos_list.append((x_m, y_m))  # 保存patch左上角的位置
            # 低清patch的位置
            pos_list_lim.append((x_l, y_l))
            # 高清patch均值
            patch_mean.append(np.mean(mim[x_m:x_m+patch_size_h, y_m:y_m+patch_size_h].reshape((h,))))
            # 待回复patch的超像素分类
            # mim_cls_list.append(mim_class[x_h, y_h])
    # ========================= 对patch进行前处理 ===========================
    feature_list, lim_list, mim_list = (np.asarray(feature_list, dtype=float), np.asarray(lim_list, dtype=float), np.asarray(mim_list, dtype=float))

    # =========================    进行实际计算   ========================================
    r_patch_list = []
    if method is "infile":
        r_patch_list = read_super_patch()
    elif method is "regression":
        r_patch_list = get_super_patch_regression_mul_with_search(feature_list, him_list, tag="mullayer36")

    # ========================      组合结果     =======================================
    r_image = combine_image_from_patch(r_patch_list, pos_list, mim, mim_list, pl=6)

    result = r_image
    result[result > 255] = 255
    result[result < 0] = 0

    # =========================     得到了提升两倍的结果：result   =================================
    # plt.imshow(result, interpolation="None", cmap=cm.gray)
    # plt.show()
    # =========================     下面需要对result进行1.5倍的提升   ==============================
    # ===================================================
    # ====================================================
    # =====================================================

    raw_img = result
    shape = raw_img.shape
    # 首先需要降低三倍
    if len(shape) == 3:
        raw_img = raw_img[:shape[0]-shape[0] % 2, :shape[1]-shape[1] % 2, :]
    else:
        raw_img = raw_img[:shape[0]-shape[0] % 2, :shape[1]-shape[1] % 2]

    # 首先提升两倍
    lim = raw_img
    mim = imresize(lim, 1.5, 'bicubic')

    lim = np.asarray(lim, dtype=float)
    mim = np.asarray(mim, dtype=float)

    feature = np.zeros((4, mim.shape[0], mim.shape[1]))

    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')

    feature[0, :, :] = convolve2d(mim, f1, mode='same')
    feature[1, :, :] = convolve2d(mim, f2, mode='same')
    feature[2, :, :] = convolve2d(mim, f3, mode='same')
    feature[3, :, :] = convolve2d(mim, f4, mode='same')

    # mim_class = super_pixel_classify(test_img, cls_num=100)

    # ================================ 将patch 和feature 分解list=================================

    feature_list = []
    him_list = []
    pos_list = []
    patch_mean = []
    lim_list = []
    mim_list = []
    pos_list_lim = []
    mim_cls_list = []

    patch_size_m = 9
    patch_size_h = 9
    patch_size_l = 6
    over_lap_l = 2
    over_lap_m = 3
    over_lap_h = 3

    size_l = lim.shape
    size_m = mim.shape

    xgrid_l = np.ogrid[0:size_l[0] - patch_size_l: patch_size_l - over_lap_l]
    ygrid_l = np.ogrid[0:size_l[1] - patch_size_l: patch_size_l - over_lap_l]
    xgrid_m = np.ogrid[0:size_m[0] - patch_size_m: patch_size_m - over_lap_m]
    ygrid_m = np.ogrid[0:size_m[1] - patch_size_m: patch_size_m - over_lap_m]

    m = patch_size_m * patch_size_m * 4
    h = patch_size_h * patch_size_h
    l = patch_size_l * patch_size_l

    # ========================     提取patch      ============================
    for x_l, x_m in zip(xgrid_l, xgrid_m):
        for y_l, y_m in zip(ygrid_l, ygrid_m):
            # 中清patch向量化
            mim_list.append(mim[x_m:x_m+patch_size_m, y_m:y_m+patch_size_m].reshape((h,)))
            # 低清patch向量化
            lim_list.append(lim[x_l:x_l+patch_size_l, y_l:y_l+patch_size_l].reshape((l,)))
            # 特征patch
            feature_list.append(feature[:, x_m:x_m+patch_size_m, y_m:y_m+patch_size_m].reshape((m,)))
            # 高清patch的位置
            pos_list.append((x_m, y_m))  # 保存patch左上角的位置
            # 低清patch的位置
            pos_list_lim.append((x_l, y_l))
            # 高清patch均值
            patch_mean.append(np.mean(mim[x_m:x_m+patch_size_h, y_m:y_m+patch_size_h].reshape((h,))))
            # 待回复patch的超像素分类
            # mim_cls_list.append(mim_class[x_h, y_h])
    # ========================= 对patch进行前处理 ===========================
    feature_list, lim_list, mim_list = (np.asarray(feature_list, dtype=float), np.asarray(lim_list, dtype=float), np.asarray(mim_list, dtype=float))

    # =========================    进行实际计算   ========================================
    r_patch_list = []
    if method is "infile":
        r_patch_list = read_super_patch()
    elif method is "regression":
        r_patch_list = get_super_patch_regression_mul_with_search(feature_list, him_list, tag="mullayer69")

    # ========================      组合结果     =======================================
    r_image = combine_image_from_patch(r_patch_list, pos_list, mim, mim_list, pl=9)

    result = r_image
    result[result > 255] = 255
    result[result < 0] = 0

    # print "均方根误差为：%d" % np.mean((him_list - r_patch_list[0])**2)**0.5
    # ========================      结果展示     =======================================
    io.imsave(output, np.asarray(result, dtype='uint8'))

    result[result == 0] = mim[result == 0]  # 处理边界
    print result.shape
    print him.shape
    print "峰值性噪比：%0.5f" % psnr(result[3:507,3:505], him[3:507,3:505])

    plt.subplot(131)
    plt.imshow(result, interpolation="None", cmap=cm.gray)
    plt.subplot(132)
    plt.imshow(him, interpolation="None", cmap=cm.gray)

    plt.show()


def super_image(method="sc", input='result/bird_GT.bmp', output="result_bird.bmp"):
    """
    处理超分辨率的输入输出
    :return:
    """
    tmp_file_name = output + '.pickle'
    '''
    ========================    得到超分辨率的输入和输出    ============================
    '''
    test_img = io.imread(input)
    shape = test_img.shape
    if len(shape) == 3:
        test_img = test_img[:shape[0] - shape[0] % 3, :shape[1] - shape[1] % 3, :]
    else:
        test_img = test_img[:shape[0] - shape[0] % 3, :shape[1] - shape[1] % 3]

    # test_img = test_img[120:240, 80:200, :]

    test_back = test_img.copy()
    feature, him, lim, mim = image_feature_generate(test_img)

    # mim_class = super_pixel_classify(test_img, cls_num=100)
    '''
     =======================    将patch 和feature 分解list    =================================
    '''
    feature_list, him_list, mim_list, pos_list = devide_into_patch(feature, him, mim)

    '''
    =========================    进行实际计算   ========================================
    '''
    r_patch_list = []
    if method is "infile":
        r_patch_list = read_super_patch(tmp_file_name)
    elif method is "regression":
        r_patch_list = get_super_patch_regression_with_search(feature_list, him_list, input_tag='11',model_tag='10', file_name=tmp_file_name)
        # r_patch_list = get_super_patch_regression_with_superpixel(feature_list, him_list, mim_cls_list, tag=1)
        # r_patch_list = get_super_patch_regression_without_search(feature_list, him_list, tag=1, lim_list=lim_list, mim_list=mim_list)
    elif method is "sc":
        r_patch_list = get_super_patch_sc(feature_list, him_list, tag=12)

    print "均方根误差为：%d" % np.mean((him_list - r_patch_list[0])**2)**0.5

    '''
    ==========================    组合结果     =======================================
    '''
    #  r_image = combine_image_from_patch(r_patch_list, pos_list, mim, mim_list)
    #  r_image = combine_image_from_patch_with_deformed(r_patch_list, pos_list, rim, mim_list=mim_list, lim_list=lim_list, patch_list=patch_list,image=image)
    result_patchmean, r_image, r_gap = combine_image_deal_with_overlap(r_patch_list, pos_list, mim, mim_list)

    result = r_image  # + rim
    result[result > 235] = 235
    result[result < 16] = 16

    result_mean = result_patchmean  # + rim
    result_mean[result_mean > 235] = 235
    result_mean[result_mean < 16] = 16

    # ========================= 组合成彩色图像 =======================
    '''
    test_back_ycbcr = np.asarray(rgb2ycbcr(test_back), dtype=float)
    p = test_back_ycbcr.copy()
    test_back_ycbcr[:, :, 0] = result_mean
    result_color = np.asarray(ycbcr2rgb(test_back_ycbcr), dtype=float)
    '''

    '''
    ========================    进行测试结果:比较降低后的结果     =============================
    '''
    compare_lim(lim, result_mean)

    '''
    ========================   使用opencv做锐化   ==============================
    '''
    result_sharp = image_sharper(result_mean)

    '''
    ========================   保存结果    =============================
    '''
    io.imsave(output, np.asarray(result, dtype='uint8'))

    '''
    ========================   误差计算   ======================================
    '''

    result[result == 0] = mim[result == 0]  # 处理边界
    print "patch直接均值峰值性噪比：%0.5f" % psnr(result_mean[3:-3, 3:-3], him[3:-3,3:-3])
    print "像素处理峰值性噪比：%0.5f" % psnr(result[3:-3, 3:-3], him[3:-3,3:-3])
    print "单独锐化峰值性噪比：%0.5f" % psnr(result_sharp[3:-3, 3:-3], him[3:-3,3:-3])
    #print "彩色图像峰值性噪比：%0.5f" % psnr(test_back_ycbcr[3:-3, 3:-3,:], p[3:-3,3:-3,:])

    '''
    ========================    结果展示   =======================================
            这里展示除了heavy-tailed特征以及梯度统计特征
    '''
    heavy_tailed_feature(him, color='red', label="result by patch mean")
    heavy_tailed_feature(result, color='black', label="result by dealing with pixel")
    heavy_tailed_feature(result_mean, color='blue', label="ground-truth")
    heavy_tailed_feature(lim, color='yellow', label="low resolution image")
    # heavy_tailed_feature(result_sharp, color='green', label="result by sharp seperately")
    plt.legend()
    plt.show()

    show_result(him, r_gap, result, result_mean, result_sharp)


if __name__ == '__main__':
    # super_image(method="infile",input='result/head_GT.bmp', output="result_head.bmp") #33.8
    # super_image(method="infile",input='result/bird_GT.bmp', output="result_bird.bmp") #35.24
     super_image(method="infile",input='result/baby_GT.bmp', output="result_baby.bmp") #35.25
    # super_image(method="infile",input='result/butterfly_GT.bmp', output="result_butterfly.bmp") #26.32
    # super_image(method="infile",input='result/woman_GT.bmp', output="result_woman.bmp")
    # super_image_mullayer(method="regression")
