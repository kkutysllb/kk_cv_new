#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-12-13 23:51
# @Desc   : 读取数据集文本标签，并生成标签列表
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

def read_and_sort_fruits(filename):
    # Read the file and extract fruit names
    with open(filename, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Sort alphabetically
    class_names.sort()
    return class_names

# # Example usage:
# filename = "data/frutis100/classname.txt"
# sorted_fruits = read_and_sort_fruits(filename)

# # Print sorted list
# for fruit in sorted_fruits:
#     print(fruit)
if __name__ == "__main__":
    file_path = os.path.join(root_dir, 'data', 'frutis100', 'classname.txt')
    sorted_fruits = read_and_sort_fruits(file_path)
    print(sorted_fruits)
