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
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

np.random.seed(333)
torch.random.manual_seed(333)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.kk_layer = nn.Sequential(
            nn.Linear(4, 200),
            nn.PReLU(),
            nn.Linear(200, 100),
            nn.PReLU(),
            nn.Linear(100, 3)
        )
        
    # def _init_weights(self):
    #     for m in self.parameters():
    #         nn.init.normal_(m)
             
    def forward(self, x):
        return self.kk_layer(x)
        
    


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(parent_dir, 'data', 'iris.csv'))
    # print(df.head())
    # input = torch.randn(8, 4)
    # print(IrisNet()(input))
    
    net = IrisNet()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=0.0011)
    
    # 使用tensorboard记录训练过程
    writer = SummaryWriter(os.path.join(parent_dir, 'runs', 'iris_net'))
    writer.add_graph(net, torch.randn(1, 4))
    
    # 训练
    total_epochs = 300
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
            writer.add_scalar('loss', loss.item(), epoch * total_batch + batch)
            writer.add_scalar('acc', acc, epoch * total_batch + batch)
            writer.close()
        
    # 测试
    x = torch.tensor(df.iloc[:, :4].values.astype(np.float32), dtype=torch.float32)
    y_pred = net(x)
    df['pred'] = y_pred.argmax(dim=-1).cpu().numpy()
    df.to_csv(os.path.join(parent_dir, 'data', 'iris_pred.csv'), index=False, sep=',', encoding='utf-8')
