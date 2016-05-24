# coding:utf-8
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchDictionaryLearning
import matplotlib.cm as cm
from sklearn.cluster import KMeans,AffinityPropagation,AgglomerativeClustering, DBSCAN

__author__ = "liangz14"


def onto_unit(x):
    """
    对patch进行统一范围
    """
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)


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


def k_means_classify(data_list, n_clusters=15, n_components=30, pca=None):
    """
        使用k-mean对patch进行分类
        list 原始数据 （num，dim）
        :n_clusters: 需要分类的数量
        :n_components: 需要使用的维度
        :return: 表示分类结果
    """
    if len(data_list[1]) > n_components:
        if pca is None:
            # 将原本的数据进行降维
            print "生成PCA进行降维"
            pca = PCA(n_components=n_components)
            pca = pca.fit(data_list)
            data_list = pca.transform(data_list)
        else:
            print "用已有的PCA进行降维"
            data_list = pca.transform(data_list)
    else:
        print "已进行降维"
    # 进行k-means聚类
    k_means = KMeans(n_clusters=n_clusters)
    k_means = k_means.fit(data_list)
    y_predict = k_means.predict(data_list)

    return y_predict, k_means, pca


def k_means_for_lib():
    """
    尝试对码本进行聚类，输出聚类后新的结果

    :return:
    """
    file_name = "2_sc_list_new_clsd_raw_full.pickle"
    sc_file = open(file_name, 'rb')
    sc_list = cPickle.load(sc_file)
    sc_file.close()
    sc_list = np.concatenate(sc_list)
    # low_dict = sc_list[:, :144].T
    high_dict = sc_list[:, 144:].T

    y_predict, _, _ = k_means_classify(high_dict.T)

    print len(y_predict)
    Dh =[]
    num = []
    y_tmp = np.asarray(y_predict, dtype=int) * 0 + 1
    for i in range(len(np.unique(y_predict))):
        num.append(np.sum(y_tmp[y_predict == i]))
    rand = np.asarray(num).argsort()  # 按照各个类别patch个数从少到多排序的类别索引

    print num
    print np.asarray(num)[rand]

    classified_patch = []

    for i in rand:
        predict_temp = y_predict == i
        classified_patch.append(high_dict[predict_temp])

    print
    for i in range(9):
        x = i % 3
        y = i / 3

        patch_show(classified_patch[i+3][:81], [0.05+x*0.31, 0.05+y*0.31, 0.3, 0.3], i)
    plt.show()


def train_sparse_coding(feature_list, patch_list, dict_size=256, transform_alpha=0.5, n_iter=50):
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
        dico = None
        X = np.concatenate((feature, patch), axis=1)

        if len(X) > 100000:
            np.random.shuffle(X)
            X = X[:90000]

        if len(X) < 5000:
            print "进入DictionaryLearning状态"
            dico = MiniBatchDictionaryLearning(batch_size=1000, transform_algorithm='lasso_lars', fit_algorithm='lars',
                                               transform_n_nonzero_coefs=5, n_components=len(X)/50,
                                               dict_init=X[:len(X)/50],
                                               n_iter=n_iter, transform_alpha=transform_alpha, verbose=10, n_jobs=-1)
        else:
            print "进入MiniBatchDictionaryLearning状态"
            dico = MiniBatchDictionaryLearning(batch_size=1000, transform_algorithm='lasso_lars', fit_algorithm='lars',
                                               transform_n_nonzero_coefs=5, n_components=len(X)/50,
                                               dict_init=X[:len(X)/50],
                                               n_iter=n_iter, transform_alpha=transform_alpha, verbose=10, n_jobs=-1)
        V = dico.fit(X).components_
        sc_list.append(V)

        file_name = "./tmp_file/_tmp_sc_list_new_clsd_raw_%d.pickle" % (i)
        sc_file = open(file_name, 'wb')
        cPickle.dump(sc_list, sc_file, 1)
        sc_file.close()

    return sc_list


def filter_patch_for_regression(feature_list, patch_list):
    """
    为了满足系数编码的条件，X需要减去均值在除以方差，Y需要减去均值；
    为了X和Y是成比例的，所以Y需要除以X的方差
    patch_list :是输入的patch列表
    feature_list :是没有经过pca或其他降维操作
    n_components：将到这一维度进行过滤
    :return:
        筛选后的：feature_list
        筛选后的：patch_list
        pca降维模型
    """
    print "样本数为：%d" % len(patch_list)

    '''
    # 减去均值
    patch_mean = np.mean(patch_list, axis=1)
    patch_mean = np.repeat([patch_mean], len(patch_list[1]), axis=0).T
    patch_list = (patch_list - patch_mean)


    fea_mean = np.mean(feature_list, axis=1)
    fea_mean = np.repeat([fea_mean], len(feature_list[1]), axis=0).T
    feature_list = (feature_list - fea_mean)
    '''
    # 定义一个极小值 消去除0误差
    eps = np.finfo(float).eps
    # X除均方差
    mes = (np.sum(feature_list ** 2.0, axis=1)) ** 0.5 + eps

    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_list = feature_list / mes_list

    # 这里就是将减去均值后的patch在除去
    mes_list = np.repeat([mes], len(patch_list[1]), axis=0).T
    patch_list = patch_list / mes_list

    return feature_list, patch_list


def filter_patch_for_sc(feature_list, patch_list):
    """
    为了满足系数编码的条件，X需要减去均值在除以方差，Y需要减去均值；
    为了X和Y是成比例的，所以Y需要除以X的方差
    patch_list :是输入的patch列表
    feature_list :是没有经过pca或其他降维操作
    n_components：将到这一维度进行过滤
    :return:
        筛选后的：feature_list
        筛选后的：patch_list
        pca降维模型
    """
    print "样本数为：%d" % len(patch_list)

    # 减去均值
    patch_mean = np.mean(patch_list, axis=1)
    patch_mean = np.repeat([patch_mean], len(patch_list[1]), axis=0).T
    patch_list = (patch_list - patch_mean)

    fea_mean = np.mean(feature_list, axis=1)
    fea_mean = np.repeat([fea_mean], len(feature_list[1]), axis=0).T
    feature_list = (feature_list - fea_mean)

    # 定义一个极小值 消 去除0误差
    eps = np.finfo(float).eps
    # X除均方差
    mes = (np.sum(feature_list ** 2.0, axis=1)) ** 0.5 + eps
    mask = mes > 0

    mes_list = np.repeat([mes], len(feature_list[1]), axis=0).T
    feature_list = feature_list / mes_list

    # 这里就是将减去均值后的patch在除去
    mes_list = np.repeat([mes], len(patch_list[1]), axis=0).T
    patch_list = patch_list / mes_list

    feature_list = feature_list[mask]
    patch_list = patch_list[mask]

    return feature_list, patch_list


def learn_for_regression(use_classify=False, use_pca=False, tag=1, n_components=30):
    """
    1.读取训练数据进行预处理
    2.根据feature计算训练patch分类进行保存
    3.
    :return:
    """

    # '''
    # ====================================计算分类并保存=======================================
    # =====================   training_data.pickle ---》class_result.pickle   ================
    train_file = open('./tmp_file/_%d_training_data.pickle' % 1, 'rb')
    #train_file = open('./tmp_file/_%d_training_data.pickle' % 3, 'rb')
    training_data = cPickle.load(train_file)
    train_file.close()

    # 得到训练向量
    patch_lib, feature_lib = training_data
    patch_lib, feature_lib = np.asarray(patch_lib, dtype=float), np.asarray(feature_lib, dtype=float)

    y_predict, k_means, pca = None, None, None
    # 数据降维
    if use_pca:
        print "使用pca降维"
        pca = PCA(n_components=n_components)
        pca = pca.fit(feature_lib)
        feature_lib = pca.transform(feature_lib)

    # 数据过滤，归一化,准备需要训练的数据
    (feature_lib, patch_lib) = filter_patch_for_regression(feature_lib, patch_lib)

    # 进行分类、并保存分类结果
    if use_classify:
        y_predict, k_means, pca = k_means_classify(feature_lib, pca=pca, n_clusters=15)   # 尝试使用patch来进行聚类
    else:
        y_predict = np.zeros((len(feature_lib),))  # 表示全部属于同一类

    # 保存分类标签结果
    with open('./tmp_file/%d_class_tag.pickle' % tag, 'wb') as f:
        cPickle.dump(y_predict, f, 1)

    with open('./tmp_file/%d_kmeans_pca_model.pickle' % tag, 'wb') as f:
        cPickle.dump((k_means, pca), f, 1)

    with open('./tmp_file/%d_training_data_normalized.pickle' % tag, 'wb') as f:
        cPickle.dump((feature_lib, patch_lib), f, 1)
    # '''

    # '''
    # ==========================读取上一步内容==================================================
    f = open('./tmp_file/%d_class_tag.pickle' % tag, 'rb')
    y_predict = cPickle.load(f)

    f = open('./tmp_file/%d_kmeans_pca_model.pickle' % tag, 'rb')
    (k_means, pca) = cPickle.load(f)

    f = open('./tmp_file/%d_training_data_normalized.pickle' % tag, 'rb')
    (feature_lib, patch_lib) = cPickle.load(f)
    # '''

    # '''
    # =========================通过分类结果将训练数据切分并保存=======================================
    # ===================== class_result.pickle ---》_training_data_classified.pickle==========
    # 将数据根据类别分解并保存
    num = []
    y_tmp = np.asarray(y_predict, dtype=int) * 0 + 1
    for i in range(len(np.unique(y_predict))):
        num.append(np.sum(y_tmp[y_predict == i]))
    rand = np.asarray(num).argsort()  # 按照各个类别patch个数从少到多排序的类别索引

    print np.asarray(num)[rand]

    classified_feature = []
    classified_patch = []
    for i in rand:
        predict_temp = y_predict == i
        classified_feature.append(feature_lib[predict_temp])
        classified_patch.append(patch_lib[predict_temp])

    # 保存分类结果
    classified_file = open('./tmp_file/%d_training_data_classified.pickle' % tag, 'wb')
    cPickle.dump((classified_feature, classified_patch), classified_file, 1)
    classified_file.close()
    # '''

    '''
    # ========================对分割后的数据、读取k-means、pca转换后进行稀疏编码并保存==============================
    classified_file = open('./tmp_file/%d_class_with_regression.pickle' % tag, 'rb')
    #classified_file = open('./tmp_file/%d_class_result.pickle' % tag, 'rb')
    (classified_feature, classified_patch, classified_error, class_tag) = cPickle.load(classified_file)

    # 由于训练实践过长，改在内部训练一个类型后马上进行保存
    sc_list = train_sparse_coding(classified_feature[:], classified_patch[:])
    sc_file = open('./tmp_file/%d_dictionary_regression.pickle' % tag, 'wb')
    cPickle.dump(sc_list, sc_file, 1)
    sc_file.close()
    # '''


def learn_for_sc(use_classify=False, use_pca=False, tag=1, n_components=30):
    """
    1.读取训练数据进行预处理
    2.根据feature计算训练patch分类进行保存
    3.
    :return:
    """

    #'''
    # ====================================计算分类并保存=======================================
    # =====================   training_data.pickle ---》class_result.pickle   ================
    train_file = open('./tmp_file/_%d_training_data.pickle' % tag, 'rb')
    training_data = cPickle.load(train_file)
    train_file.close()

    # 得到训练向量
    patch_lib, feature_lib = training_data
    patch_lib, feature_lib = np.asarray(patch_lib, dtype=float), np.asarray(feature_lib[0:30000], dtype=float)
    print len(patch_lib)
    y_predict, k_means, pca = None, None, None
    # 数据降维
    if use_pca:
        print "使用pca降维"
        pca = PCA(n_components=n_components)
        pca = pca.fit(feature_lib)
        feature_lib = pca.transform(feature_lib)

    # 数据过滤，归一化,准备需要训练的数据
    (feature_lib, patch_lib) = filter_patch_for_sc(feature_lib, patch_lib)

    # 进行分类、并保存分类结果
    if use_classify:
        y_predict, k_means, pca = k_means_classify(feature_lib, pca=pca)   # 尝试使用patch来进行聚类
    else:
        y_predict = np.zeros((len(feature_lib),))  # 表示全部属于同一类

    # 保存分类标签结果
    with open('./tmp_file/%d_class_tag.pickle' % tag, 'wb') as f:
        cPickle.dump(y_predict, f, 1)

    with open('./tmp_file/%d_kmeans_pca_model.pickle' % tag, 'wb') as f:
        cPickle.dump((k_means, pca), f, 1)

    with open('./tmp_file/%d_training_data_normalized.pickle' % tag, 'wb') as f:
        cPickle.dump((feature_lib, patch_lib), f, 1)
    # '''

    #'''
    # ==========================读取上一步内容==================================================
    f = open('./tmp_file/%d_class_tag.pickle' % tag, 'rb')
    y_predict = cPickle.load(f)

    f = open('./tmp_file/%d_kmeans_pca_model.pickle' % tag, 'rb')
    (k_means, pca) = cPickle.load(f)

    f = open('./tmp_file/%d_training_data_normalized.pickle' % tag, 'rb')
    (feature_lib, patch_lib) = cPickle.load(f)
    # '''

    #'''
    # =========================通过分类结果将训练数据切分并保存=======================================
    # ===================== class_result.pickle ---》_training_data_classified.pickle==========
    # 将数据根据类别分解并保存
    num = []
    y_tmp = np.asarray(y_predict, dtype=int) * 0 + 1
    for i in range(len(np.unique(y_predict))):
        num.append(np.sum(y_tmp[y_predict == i]))
    rand = np.asarray(num).argsort()  # 按照各个类别patch个数从少到多排序的类别索引

    print np.asarray(num)[rand]

    classified_feature = []
    classified_patch = []
    for i in rand:
        predict_temp = y_predict == i
        classified_feature.append(feature_lib[predict_temp])
        classified_patch.append(patch_lib[predict_temp])

    # 保存分类结果
    classified_file = open('./tmp_file/%d_training_data_classified.pickle' % tag, 'wb')
    cPickle.dump((classified_feature, classified_patch), classified_file, 1)
    classified_file.close()
    # '''

    '''
    # ========================对分割后的数据、读取k-means、pca转换后进行稀疏编码并保存==============================
    classified_file = open('./tmp_file/%d_class_with_regression.pickle' % tag, 'rb')
    # classified_file = open('./tmp_file/%d_class_result.pickle' % tag, 'rb') # 为使用回归分类，直接进行回归
    (classified_feature, classified_patch, classified_error, class_tag) = cPickle.load(classified_file)

    # 由于训练实践过长，改在内部训练一个类型后马上进行保存
    sc_list = train_sparse_coding(classified_feature[:], classified_patch[:])
    sc_file = open('./tmp_file/%d_dictionary_regression.pickle' % tag, 'wb')
    cPickle.dump(sc_list, sc_file, 1)
    sc_file.close()
    # '''


def learn_for_regression_mullayer(use_classify=False, use_pca=False, tag="mullayer36", n_components=30):
    """
    1.读取训练数据进行预处理
    2.根据feature计算训练patch分类进行保存
    3.
    :return:
    """

    # '''
    # ====================================计算分类并保存=======================================
    # =====================   training_data.pickle ---》class_result.pickle   ================
    train_file = open('./tmp_file/_%s_training_data.pickle' % tag, 'rb')
    training_data = cPickle.load(train_file)
    train_file.close()

    # 得到训练向量
    patch_lib, feature_lib = training_data
    patch_lib, feature_lib = np.asarray(patch_lib, dtype=float), np.asarray(feature_lib, dtype=float)

    y_predict, k_means, pca = None, None, None

    # 数据过滤，归一化,准备需要训练的数据,需要在降维之前先进性一次归一化
    # (feature_lib, patch_lib) = filter_patch_for_regression(feature_lib, patch_lib)
    # 数据降维
    if use_pca:
        print "使用pca降维"
        pca = PCA(n_components=n_components)
        pca = pca.fit(feature_lib)
        feature_lib = pca.transform(feature_lib)

    # 数据过滤，归一化,准备需要训练的数据
    (feature_lib, patch_lib) = filter_patch_for_regression(feature_lib, patch_lib)

    # 进行分类、并保存分类结果
    if use_classify:
        y_predict, k_means, pca = k_means_classify(feature_lib, pca=pca, n_clusters=15)   # 尝试使用patch来进行聚类
    else:
        y_predict = np.zeros((len(feature_lib),))  # 表示全部属于同一类

    # 保存分类标签结果
    with open('./tmp_file/%s_class_tag.pickle' % tag, 'wb') as f:
        cPickle.dump(y_predict, f, 1)

    with open('./tmp_file/%s_kmeans_pca_model.pickle' % tag, 'wb') as f:
        cPickle.dump((k_means, pca), f, 1)

    with open('./tmp_file/%s_training_data_normalized.pickle' % tag, 'wb') as f:
        cPickle.dump((feature_lib, patch_lib), f, 1)
    # '''

    # '''
    # ==========================读取上一步内容==================================================
    f = open('./tmp_file/%s_class_tag.pickle' % tag, 'rb')
    y_predict = cPickle.load(f)

    f = open('./tmp_file/%s_kmeans_pca_model.pickle' % tag, 'rb')
    (k_means, pca) = cPickle.load(f)

    f = open('./tmp_file/%s_training_data_normalized.pickle' % tag, 'rb')
    (feature_lib, patch_lib) = cPickle.load(f)
    # '''

    # '''
    # =========================通过分类结果将训练数据切分并保存=======================================
    # ===================== class_result.pickle ---》_training_data_classified.pickle==========
    # 将数据根据类别分解并保存
    num = []
    y_tmp = np.asarray(y_predict, dtype=int) * 0 + 1
    for i in range(len(np.unique(y_predict))):
        num.append(np.sum(y_tmp[y_predict == i]))
    rand = np.asarray(num).argsort()  # 按照各个类别patch个数从少到多排序的类别索引

    print np.asarray(num)[rand]

    classified_feature = []
    classified_patch = []
    for i in rand:
        predict_temp = y_predict == i
        classified_feature.append(feature_lib[predict_temp])
        classified_patch.append(patch_lib[predict_temp])

    # 保存分类结果
    classified_file = open('./tmp_file/%s_training_data_classified.pickle' % tag, 'wb')
    cPickle.dump((classified_feature, classified_patch), classified_file, 1)
    classified_file.close()
    # '''

    '''
    # ========================对分割后的数据、读取k-means、pca转换后进行稀疏编码并保存==============================
    classified_file = open('./tmp_file/%d_class_with_regression.pickle' % tag, 'rb')
    #classified_file = open('./tmp_file/%d_class_result.pickle' % tag, 'rb')
    (classified_feature, classified_patch, classified_error, class_tag) = cPickle.load(classified_file)

    # 由于训练实践过长，改在内部训练一个类型后马上进行保存
    sc_list = train_sparse_coding(classified_feature[:], classified_patch[:])
    sc_file = open('./tmp_file/%d_dictionary_regression.pickle' % tag, 'wb')
    cPickle.dump(sc_list, sc_file, 1)
    sc_file.close()
    # '''


# learn_for_sc(use_classify=True, use_pca=True, tag=100)
learn_for_regression(use_classify=True, use_pca=True, tag=1)
# learn_for_regression_mullayer(use_classify=True, use_pca=True, tag="mullayer69")
#learn_for_regression_mullayer(use_classify=True, use_pca=True, tag="mullayer36")

