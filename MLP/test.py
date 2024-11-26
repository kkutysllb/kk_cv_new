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

if torch.backends.mps.is_available():
    print("MPS is available")
    device = "mps"
elif torch.cuda.is_available():
    print("CUDA is available")
    device = "cuda"
else:
    print("MPS is not available")
    device = "cpu"