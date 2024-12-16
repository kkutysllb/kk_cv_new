#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-10-14 21:15
# @Desc   : 魔塔社区数据集测试
# --------------------------------------------------------
"""

from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

if __name__ == "__main__":
    # train_dataset = MsDataset.load(
    #     'mini_imagenet100', namespace='tany0699', subset_name='default', split='train', download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
    # )
    
    # print(next(iter(train_dataset)))

    def read_class_names(filename='data/mini_imagenet100/classname.txt'):
        with open(filename, 'r') as f:
            # 读取所有行，去除每行首尾空格，过滤掉空行
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        return class_names
    
    print(read_class_names())
    print(len(read_class_names()))