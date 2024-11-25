#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-11-25 10:15
# @Desc   : torch实现iris数据集分类
# --------------------------------------------------------
"""
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

torch.random.manual_seed(333)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(4, 20))
        self.bias1 = nn.Parameter(torch.zeros(1, 20))
        self.weight2 = nn.Parameter(torch.randn(20, 10))
        self.bias2 = nn.Parameter(torch.zeros(1, 10))
        self.weight3 = nn.Parameter(torch.randn(10, 3))
        self.bias3 = nn.Parameter(torch.zeros(1, 3))
        self._init_weights()
        
    def _init_weights(self):
        for m in self.parameters():
            nn.init.normal_(m)
             
    def forward(self, x):
        x = x @ self.weight1 + self.bias1
        x = F.relu(x)
        x = x @ self.weight2 + self.bias2
        x = F.relu(x)
        return x @ self.weight3 + self.bias3
    
    
# class Accuracy(nn.Module):
#     def __init__(self):
#         super(Accuracy, self).__init__()
        
#     def forward(self, y_pred, y):
#         y_pred_dim = y_pred.dim()
#         y_dim = y.dim()
#         if y_pred_dim == y_dim:
#             pass
#         elif y_pred_dim == y_dim + 1:
#             y_pred = y_pred.argmax(dim=-1)
#         else:
#             raise ValueError("y_pred and y must have the same dimension")
#         y_pred = y_pred.to(y.dtype)
#         correct = (y_pred.argmax(dim=-1) == y).float()
#         return torch.mean(correct)  
    

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(parent_dir, 'data', 'iris.csv'))
    # print(df.head())
    # input = torch.randn(8, 4)
    # print(IrisNet()(input))
    
    net = IrisNet()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    
    # 训练
    total_epochs = 200
    batch_size = 16
    total_batch = 1 if batch_size >= len(df) else len(df) // batch_size
    
    for epoch in range(total_epochs):
        batch_idx = np.random.permutation(len(df))
        for batch in range(total_batch):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            if batch == total_batch - 1:
                end_idx = len(df)
            _df = df.loc[batch_idx[start_idx:end_idx]]
            x = torch.tensor(_df.iloc[:,:4].values.astype(np.float32), dtype=torch.float32)
            y = torch.tensor(_df.iloc[:, 4].values.astype(int), dtype=torch.long)
        
            # 前向传播
            y_pred = net(x)
            loss = criterion(y_pred, y)
            acc = accuracy_score(y_pred.argmax(dim=-1).cpu().numpy(), y.cpu().numpy())
        
            # 反向传播
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
            print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch+1}/{total_batch}, Loss: {loss.item():.3f}, Acc: {acc:.3f}")
        
    # 测试
    x = torch.tensor(df.iloc[:, :4].values.astype(np.float32), dtype=torch.float32)
    y_pred = net(x)
    df['pred'] = y_pred.argmax(dim=-1).cpu().numpy()
    df.to_csv(os.path.join(parent_dir, 'data', 'iris_pred.csv'), index=False, sep=',', encoding='utf-8')
