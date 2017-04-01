# coding:utf-8
import cv2
import dlib
"""
这里用来做放射变换的采样，由于目前的patch使用的都是方形的变化，所以patch的内容有限
我这里尝试使用不规则的patch来进行描述。patch的形状是平行四边形。
这里采样的标准是边界经可能的在patch的中心。而不再patch的边缘。==》二阶梯度的质心在patch的中间
"""
class affine_trans(object):
    """
    必须能够进行正反变化
    """
    pass

