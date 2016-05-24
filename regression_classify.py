# coding:utf-8
import cPickle
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

__author__ = 'liangz14'


def train_regression(feature_list, patch_list):
    """
    使用mini batch训练稀疏编码
    #feature_list 表示要训练的特征的列表
    #patch_list 表示结果patch的列表

    :return sc_list
    """
    sc_list = []
    i = 0
    for feature, patch in zip(feature_list, patch_list):
        i = i + 1
        '''
        由于组合数值大小比例的问题，稀疏编码可能忽略较小的特征,下面的×10需要用别的特征归一化方法代替
        相关性越大，则每个向量都是有用的，所以需要更长的时间进行训练。
        '''
        X = np.concatenate((feature, patch), axis=1)

        if len(X) > 100000:
            np.random.shuffle(X)
            X = X[:90000]

        coef_ = regression_ridge(feature, patch)
        sc_list.append(coef_)
        '''
        file_name = "./tmp_file/_tmp_regression_list_new_clsd_raw_%d.pickle" % (i)
        sc_file = open(file_name, 'wb')
        cPickle.dump(sc_list, sc_file, 1)
        sc_file.close()
        '''
    return sc_list


def regression_ols(x, y):
    clf = linear_model.LinearRegression(n_jobs=-1)
    clf.fit(x, y)
    return clf.coef_


def regression_ridge(x, y):
    clf = linear_model.Ridge(alpha=.5)
    clf.fit(x, y)
    return clf.coef_


def err(a, b):
    """
    均方根误差
    :param a:
    :param b:
    :return:
    """
    return np.mean((a-b)**2, axis=0)**0.5


def classify_by_regression(iter_nums=20, class_num=20000, tag='1', dele=2000.00, theta=0.0, err_del=1.0):

    # ======================== 读取初始的分类结果文件 ==============================
    classified_file = open('./tmp_file/%s_training_data_classified.pickle' % tag, 'rb')
    (classified_feature, classified_patch) = cPickle.load(classified_file)

    # -------------------------------------------------------------------------
    # --------------------------------   循环开始   ----------------------------
    # 定义噪声时为了保存噪声数据，和分离除新的分类
    classified_error = None
    noise_feature = []
    noise_patch = []
    error_mean = []
    for i in range(iter_nums):
        print "=========================== 第%d次循环数据 ===========================" % i
        # 有一个初始的分类结果：(classified_feature, classified_patch)
        # 将噪声加入某一个 分类中，或者独立成一个新的分类
        if len(noise_feature) is not 0:
            has_new_class = False
            if i % 5 == 0 and len(np.concatenate(noise_feature)) > class_num:
                noise_feature = np.asarray(np.concatenate(noise_feature), dtype=float)
                noise_patch = np.asarray(np.concatenate(noise_patch), dtype=float)
                classified_feature.append(noise_feature)
                classified_patch.append(noise_patch)
                has_new_class = True
                print "分类增加"
            else:
                for j in range(len(classified_feature)):
                    classified_feature[j] = np.concatenate((classified_feature[j], noise_feature[j]))
                    classified_patch[j] = np.concatenate((classified_patch[j], noise_patch[j]))
                    pass

            # 如果有特征大于一个值，则需要去掉这个回归函数,把数据分配到别的回归中去 / 直接去掉这些
            mask = error_mean < dele
            if has_new_class:
                mask = np.hstack((mask, np.array([True])))
            if False in mask:
                print "分类减少"
            classified_feature = np.asarray(classified_feature)[mask].tolist()
            classified_patch = np.asarray(classified_patch)[mask].tolist()

            noise_feature = []
            noise_patch = []

        # ========================  对分类结果进行回归 ===============================================
        regression_list = train_regression(classified_feature[:], classified_patch[:])

        # ========================  将所有数据进行合并  ===============================================
        feature_matrix = np.asarray(np.concatenate(classified_feature), dtype=float)
        patch_matrix = np.asarray(np.concatenate(classified_patch), dtype=float)

        # ========================  计算所有数据对于每一个回归器的回归误差 =======================================
        # 根据结果误差重新分类得到：(classified_feature, classified_patch)
        tmp_result_err = []

        for tmp_regression in regression_list:
            regression_result_tmp = np.dot(tmp_regression, feature_matrix.T)
            # 计算两种误差
            tmp_result_err.append(err(regression_result_tmp, patch_matrix.T))

        # 装换成矩阵方便计算
        tmp_result_err = np.asarray(tmp_result_err)

        # 计算每个分类最小的回归误差的那一类
        argmin_list = np.argmin(tmp_result_err, axis=0)

        classified_feature = []
        classified_patch = []
        classified_error = []
        lens = []
        lens_noi = []
        # 统计总误差
        error_all = []
        error_list = []
        num_list = []
        sum_list = []
        # ======================    通过误差的大小重新分配回归结果 ==================
        for mins in range(len(tmp_result_err)):
            mask = mins == argmin_list

            error_tmp = tmp_result_err[mins][mask]

            # 提取第i类当中，最小分类误差的特征和patch
            feature_tmp = feature_matrix[mask]
            patch_tmp = patch_matrix[mask]

            # 在每个类的内部按照特征重新排序
            argmin_tmp = np.argsort(error_tmp)
            error_list.append(error_tmp)

            sum_list.append(np.sum(error_tmp))
            num_list.append(len(error_tmp))

            # 误差从小到大排列
            feature_tmp = feature_tmp[argmin_tmp]
            patch_tmp = patch_tmp[argmin_tmp]
            error_tmp = error_tmp[argmin_tmp]
            # 在这里进行重新分配
            size = np.int(len(feature_tmp)*theta)

            # 计算新的mask 用于剔除噪声
            cls_mask = np.asarray([index < size or er <= err_del for index, er in zip(range(len(feature_tmp))
                            , error_tmp)])
            noise_mask = np.asarray([(not k) for k in cls_mask])

            classified_feature.append(feature_tmp[cls_mask])
            classified_patch.append(patch_tmp[cls_mask])
            classified_error.append(error_tmp[cls_mask])
            #保存每个patch对应的误差

            # 提取第i类当中，噪声patch和特征patch，将后百分之30作为误差项
            noise_feature.append(feature_tmp[noise_mask])
            noise_patch.append(patch_tmp[noise_mask])

            error_all.append(np.mean(error_tmp))

            # 这里为了看一下每个类别里边patch的个数
            lens.append(len(classified_patch[-1]))
            lens_noi.append(len(noise_patch[-1]))

        # hist_show(error_list, len(error_list))
        print "本次总误差为%f" % np.mean(np.asarray(error_all))
        print "各个分类中数据个数："
        print lens
        print "数据总数：%d" % np.sum(lens)
        print "各个分类中评定为噪声的个数："
        print lens_noi
        print "噪声总数：%d" % np.sum(lens_noi)
        print "本次迭代中每个类别中的平均误差："
        error_mean = np.asarray([1.0*a/b for a, b in zip(sum_list, num_list)])
        k = ["%0.4f" % i for i in error_mean]
        print k
    # -------------------------------   循环结束   ------------------------------
    # --------------------------------------------------------------------------

    # ======================== 为每一个分类添加类标================================
    class_tag = []
    for i in range(len(classified_feature)):
        class_tag.append(np.zeros((len(classified_feature[i]),))+i)
    # print class_tag
    # ======================== 保存最终的分类结果文件 ==============================
    classified_file = open('./tmp_file/%s_class_with_regression.pickle' % tag, 'wb')
    cPickle.dump((classified_feature, classified_patch, classified_error, class_tag), classified_file, 1)

    # ======================== 保存结果文件对应的回归模型 ============================
    regression_file = open('./tmp_file/%s_regression_ridge.pickle' % tag, 'wb')
    cPickle.dump(regression_list, regression_file, 1)
    regression_file.close()


def hist_show(error_list, l):
    x = np.int(np.sqrt(l))+1
    y = x
    for i in range(x):
        for j in range(y):
            ax = plt.axes([1.0*i/x+0.05, 1.0*j/y+0.05, 0.75/x, 0.75/y])
            if i*x+j >= len(error_list):
                break
            ax.hist(error_list[i*x+j], bins=100)
    plt.show()

classify_by_regression(tag="3")
#classify_by_regression(tag="mullayer36")
# classify_by_regression(tag="mullayer69")
