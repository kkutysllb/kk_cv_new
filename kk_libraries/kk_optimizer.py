#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb


import torch
import numpy as np


def kk_sgd(params, lr, batch_size):
    """定义随机梯度下降法SGD优化器"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


class kk_SGD:
    """随机梯度下降法"""

    def __init__(self, lr) -> None:
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class kk_Momentum:
    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            #对v形状与对应参数一致，全置为0
            for key, val in params.items():
                self.v[key] = torch.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class kk_Nestrov:
    """Nestrov动量算法"""

    def __init__(self, lr=0.01, momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = torch.zeros_like(value)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class kk_AdaGrad:
    """AdaGrad优化算法"""

    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = torch.zeros_like(value)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)


class kk_RMSProp:
    """RMSprop优化算法"""

    def __init__(self, lr=0.01, decay_rate=0.9) -> None:
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = torch.zeros_like(value)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)


class kk_Adam:
    """Adam优化算法"""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.v = None
        self.m = None

    def update(self, params, grads):
        if self.v is None:
            self.v, self.m = {}, {}
            for key, value in params.items():
                self.m[key] = torch.zeros_like(value)
                self.v[key] = torch.zeros_like(value)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key, value in params.items():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-8)


class kk_Yogi:
    """Yogi优化算法"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = torch.zeros_like(val)
                self.v[key] = torch.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * torch.sign(grads[key] ** 2 - self.v[key]) * grads[key] ** 2
            # sqr[:] = sqr + (1 - beta2) * torch.sign(torch.square(param.grad) - sqr) * torch.square(param.grad)
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-8)


class kk_Adadelta:
    """Adadelta优化算法"""

    def __init__(self, rho=0.9, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.Eg = None  # 梯度均值
        self.Edx = None  # 参数更新均值

    def update(self, params, grads):
        if self.Eg is None:
            self.Eg = {}
            self.Edx = {}
            # I初始化Eg和Edx，使之形状与模型参数一致
            for key, val in params.items():
                self.Eg[key] = torch.zeros_like(val)
                self.Edx[key] = torch.zeros_like(val)

        for key in params.keys():
            # 更新Eg
            self.Eg[key] = self.rho * self.Eg[key] + (1 - self.rho) * grads[key] ** 2
            # 计算更新量
            delta_x = - (torch.sqrt(self.Edx[key] + self.epsilon) / torch.sqrt(self.Eg[key] + self.epsilon)) * grads[
                key]
            # 更新参数
            params[key] += delta_x
            # 更新Edx
            self.Edx[key] = self.rho * self.Edx[key] + (1 - self.rho) * delta_x ** 2
