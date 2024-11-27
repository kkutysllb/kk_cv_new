#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from matplotlib.colors import ListedColormap
from IPython import display
from datetime import datetime

"""
自定义绘图方法
"""


def kk_plot_train_eval_curve(num_epochs, train_losses, train_accs, test_losses, test_accs, titles='Train and Evaluate Curve'):
    """绘制训练评估曲线"""
    plt.rcParams['font.sans-serif'] = ['Arial unicode MS']  # 用于正常显示中文标签
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号-

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs), train_losses, '-', label='Train Loss')
    plt.plot(np.arange(num_epochs), test_losses, 'm--', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(titles)

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_epochs), train_accs, 'g-.', label='Train Acc')
    plt.plot(np.arange(num_epochs), test_accs, 'r:', label='Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title(titles)

    plt.show()



def kk_plot_decision_boundary(model, X, y):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    light_cmap = ListedColormap(['royalblue', 'lightcoral'])

    plt.pcolormesh(xx, yy, Z, cmap=light_cmap, alpha=0.7)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of This Model')
    plt.show()


def kk_use_svg_display():
    """指定matplotlib软件包输出svg图表以获得更清晰的图像。"""
    plt.rcParams['font.sans-serif'] = ['Arial unicode MS']  # 用于正常显示中文标签
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号-
    backend_inline.set_matplotlib_formats('svg')


def kk_set_figsize(figsize=(6.5, 3.5)):
    """设置matplotlib图表的大小"""
    kk_use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def kk_set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib图表的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def kk_plot(X, Y=None, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', axes=None,
            fmts=['-', 'm--', 'g-.', 'r:'], figsize=(6.5, 3.5), legend=None, titles=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    kk_set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    if titles is not None:
        axes.set_title(titles)
    kk_set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class kk_Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(8.5, 5.5), title='Example'):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        kk_use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: kk_set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.title = title

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        if self.title:
            self.axes[0].set_title(self.title)
        display.display(self.fig)
        display.clear_output(wait=True)


def kk_datetime(year, month, day):
    """处理时间数据, 返回画图的格式数据"""
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(year, month, day)]
    return [datetime.strptime(date, '%Y-%m-%d') for date in dates]


def kk_multi_plot(x, y, nrows, ncols, xlabels, ylabels, titles, figsize=(10, 10)):
    """绘制日期坐标的多图"""
    # 设置默认格式
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号-

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # 设置布局
    fig.autofmt_xdate(rotation=45)

    # 转换为列表
    axes = [ax for axs in axes for ax in axs]

    def subplot_plot(axes, x, y, xlabel, ylabel, title):
        """绘制子图"""
        axes.plot(x, y)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

    for y, xlabel, ylabel, title, ax in zip(y, xlabels, ylabels, titles, axes):
        subplot_plot(ax, x, y, xlabel, ylabel, title)

    plt.tight_layout(pad=2)
    plt.show()


def kk_plot_fitting_curve(true_data, pred_data, xlabel=None, ylabel=None,
                          xlim=None, ylim=None, xscale='linear', yscale='linear',
                          figsize=(8.5, 5.5), titles=None):
    """绘制拟合曲线"""
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=figsize)
    # 真实值
    plt.plot(true_data['date'], true_data['actual'], '-', label='True Data')
    # 预测值
    plt.plot(pred_data['date'], pred_data['prediction'], 'mo', label='Predicted Data')
    fig.autofmt_xdate(rotation=45)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titles)

    plt.legend()
    plt.show()
