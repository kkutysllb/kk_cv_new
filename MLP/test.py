#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-11-25 10:15
# @Desc   : 测试
# --------------------------------------------------------
"""
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)
print(iris)
