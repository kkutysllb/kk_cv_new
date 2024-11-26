#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-11-26 11:16
# @Desc   : 提取每个卷积、池化、激活后feature map可视化
# --------------------------------------------------------
"""
from torchvision import models,datasets, transforms
from PIL import Image
import sys
import os
import torch
import torchvision

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

class VggHooks:
    def __init__(self, model):
        self.imgs = {}
        for idx in range(len(model.features)):
            model.features[idx].register_forward_hook(self.create_hooks(idx))

    def create_hooks(self, idx):
        def hook_fn(module, input, output):
            self.imgs[idx] = output.detach().cpu().clone()
        return hook_fn

weights_path = os.path.join(parent_dir, "outputs", "best_weights")
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

feature_path = os.path.join(parent_dir, "outputs", "features")
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

if __name__ == "__main__":
    model = models.vgg16(pretrained=True)
    # 注册hook函数，实现前向传播每一层的feature map提取
    # print(model)
    image_file = os.path.join(parent_dir, "data", "animals", "butterfly.jpg")
    img = Image.open(image_file)
    name = image_file.split("/")[-1].split(".")[0]
    img = img.convert("RGB")
    img.show()

    torch.save(model, os.path.join(parent_dir, "outputs", "best_weights", "vgg16.pth"))
    vgg_hooks = VggHooks(model)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img = transform(img).unsqueeze(0)
    
    print(img.shape)

    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        output = model(img)
        print(output.argmax(-1))

    
    print(name)
    feature_name = os.path.join(feature_path, name)
    if not os.path.exists(feature_name):
        os.makedirs(feature_name)
    ii = transforms.Resize((50, 60))
    for idx in range(len(vgg_hooks.imgs)):
        torchvision.utils.save_image(ii(vgg_hooks.imgs[idx].permute(1, 0, 2, 3)), os.path.join(parent_dir, "outputs", "features", name, f"{name}_{idx}.png"), 
                                     nrow=8)
    
