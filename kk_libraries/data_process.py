#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb


from kk_libraries.kk_dataprocess import kk_get_data_mean_stdv2, kk_split_train_test

if __name__ == '__main__':
    root_path = '../data/data_cat_dog'

    # 数据集划分训练集和测试集
    kk_split_train_test(root_path)

    # 获取自有数据均值、方差和标准差
    kk_get_data_mean_stdv2(root_path)

