#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-12-08 21:15
# @Desc   : 数据集处理
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
from torchvision import datasets
from torch.utils.data import DataLoader

class kk_Dataset:
    def __init__(self, root_dir, batch_size, shuffle, num_workers, transform):
        super(kk_Dataset, self).__init__()
        self.root_dir = root_dir
        # 数据集加载
        classes, class_to_idx = datasets.folder.find_classes(self.root_dir)
        images_path = datasets.ImageFolder.make_dataset(
            directory=self.root_dir,
            class_to_idx=class_to_idx,
            extensions=('.jpg', '.jpeg', '.png'),
        )
        self.image_path = [img[0] for img in images_path]
        # 基于torchvision的API构建遍历对象
        self.dataset = datasets.ImageFolder(
            root=self.root_dir,
            transform=transform
        )
        prefetch_factor = 2 if num_workers == 0 else num_workers * batch_size
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        
    def __len__(self):
        return len(self.dataset.imgs)
    
    def __iter__(self):
        for data in self.loader:
            yield data


if __name__ == "__main__":
    print(root_dir)