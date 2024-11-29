#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb

import copy
from datetime import datetime
import os
import torch
import numpy as np
import pandas as pd
import time
import platform
import warnings
from kk_libraries.kk_plots import kk_Animator, kk_plot_train_eval_curve, kk_plot
import glob
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def kk_random_seed():
    """设置随机种子"""
    torch.manual_seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.deterministic = True

def kk_set_cache_dir_models():
    """设置模型下载的默认保存目录"""
    os.environ['TORCH_HOME'] = r'D:\torch_models'


def warn_with_ignore(message, category, *args, **kwargs):
    """屏蔽pycharm的打印告警"""
    if "Using a target size" not in str(message):
        warnings.warn(message, category, *args, **kwargs)


def get_device():
    """获取系统信息，完成设备获取"""
    os_name = platform.system()
    # 判断系统类型，并指定相关设备
    if os_name == 'Darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def get_env_info():
    print('torch version: ', torch.__version__)
    print('cuda version: ', torch.version.cuda)
    print('cudnn version: ', torch.backends.cudnn.version())


def get_newest_file_with_prefix(folder_path, prefix):
    # 获取文件夹下所有以指定前缀开头的文件的路径
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, f)) and f.startswith(prefix)]
    # 如果没有符合条件的文件，返回 None
    if not files:
        print('没有生成的文件')
        return None, None

    # 获取每个文件的生成时间
    file_times = [(file, os.path.getctime(file)) for file in files]
    # 找到最新的文件
    newest_file = max(file_times, key=lambda x: x[1])

    # 转换时间戳为人类可读的格式
    newest_file_time = datetime.fromtimestamp(newest_file[1])
    print(f'最新生成的文件为: {newest_file[0]}, 生成时间为: {newest_file_time}')
    return newest_file[0], newest_file_time


def get_best_model_dict(model, folder_path, prefix, best_file_name=None):
    """获取已保存的最佳模型参数"""
    if best_file_name is None:
        # 获取最新的保存参数
        best_file, _ = get_newest_file_with_prefix(folder_path, prefix)
    else:
        # 获取指定保存的参数
        best_file = best_file_name
    # 加载最佳参数
    checkpoint = torch.load(best_file)
    model.load_state_dict(checkpoint['state_dict'])
    return model


"""
运行时间基准测试定义
"""


class kk_Timer:
    def __init__(self):
        """记录多次运行时间"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时，并将运行时间记录到列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return np.mean(self.times)

    def sum(self):
        """返回时间总和"""
        return np.sum(self.times)

    def cumsum(self):
        """返回累计时间, 列表形式"""
        return np.array(self.times).cumsum().tolist()

    def __repr__(self):
        return "kk_Timer()"


'''
定义一个累加器, 随着时间推移完成变量的累加, n带边要累加n个变量
'''


class kk_Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


'''
softmax回归器相关定义
'''


def kk_train_epoch_softmax(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = kk_Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), kk_accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def kk_train_softmax(net, train_iter, test_iter, loss, num_epochs, updater):
    """可视化训练模型"""
    animator = kk_Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0001, 0.9999],
                           legend=['train loss', 'train acc', 'test_cat_dog acc'], figsize=(8.5, 5.5))
    for epoch in range(num_epochs):
        train_metrics = kk_train_epoch_softmax(net, train_iter, loss, updater)
        test_acc = kk_evaluate_accuracy(net, test_iter, loss)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


'''
自定义损失函数集合
'''


def kk_mse(y_hat, y_true):
    '''定义损失函数MSELoss'''
    return (y_hat - y_true.reshape(y_hat.shape)) ** 2 / 2


def kk_cross_entropy(y_hat, y):
    """定义cross_entropy损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])


def kk_evaluate_poly_loss(net, data_iter, loss):
    """评定给定数据集上模型的损失"""
    # 记录损失的总和和样本数量
    metrics = kk_Accumulator(2)

    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metrics.add(l.sum(), l.numel())
    return metrics[0] / metrics[1]


def convert_seconds(seconds):
    """秒数转换为时分秒格式"""
    hours, remainder = divmod(seconds, 3600)  # 1小时=3600秒
    minutes, seconds = divmod(remainder, 60)  # 1分钟=60秒
    return hours, minutes, seconds


def kk_accuracy(y_hat, y):
    """定义分类精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def kk_train_accuracy(net, data_iter, criterion, optimizer, device=None):
    """在设备上训练"""
    if isinstance(net, torch.nn.Module):
        if not device:
            device = next(net.parameters()).device
        net.to(device)
    net.train()
    metrics = kk_Accumulator(3)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = net(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            metrics.add(loss * X.shape[0], kk_accuracy(y_pred, y), X.shape[0])
        # # 手动清理缓存
        # torch.cuda.empty_cache()
    return metrics[0] / metrics[2], metrics[1] / metrics[2], metrics[2]


def kk_evaluate_accuracy(net, data_iter, criterion, device=None):
    """GPU模式的评估方法"""
    if isinstance(net, torch.nn.Module):
        if not device:
            device = next(iter(net.parameters())).device
    net.to(device)
    net.eval()
    metric = kk_Accumulator(3)  # 正确预测数，预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]  # BERT微调所需
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(criterion(net(X), y) * X.shape[0], kk_accuracy(net(X), y), X.shape[0])
            # # 手动清理缓存
            # torch.cuda.empty_cache()
    return metric[0] / metric[2], metric[1] / metric[2], metric[2]


def kk_predict_accuracy(net, data_iter, device=None):
    """模型预测"""
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    net.to(device)
    metric = kk_Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(kk_accuracy(net(X), y), X.shape[0])
    return metric[0] / metric[1]


def kk_static_train_evaluate(model, train_iter, valid_iter, criterion, optimizer, num_epochs,
                             device=None, titles='Examples', print_epochs=1, scheduler=None,
                             model_name='kkutys'):
    """
    模型训练和评估，可视化展示训练曲线，训练精度和测试精度
    :param model: 模型
    :param train_iter: 训练集数据，一个生成器
    :param valid_iter: 验证集数据，一个生成器
    :param criterion: 损失函数loss
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param num_epochs: 训练轮次
    :param device: 设备类型
    :param titles: 绘图标题
    :param print_epochs: 打印轮次设置
    :param model_name: 模型名称
    :return: train_losses, train_accs, test_accs
    """

    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
    state = {}
    total_trains, total_valids = 0., 0.
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]

    # 构建一个计时器统计训练时长
    timer = kk_Timer()
    timer.start()
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc, num_examples_train = kk_train_accuracy(model, train_iter, criterion, optimizer, device)
        # 记录训练信息
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        total_trains += num_examples_train
        # 评估
        valid_loss, valid_acc, num_examples_valid = kk_evaluate_accuracy(model, valid_iter, criterion, device)
        # 记录评估信息
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        total_valids += num_examples_valid

        # 按照预先设置打印轮次打印记录
        if (epoch + 1) % print_epochs == 0 or epoch == num_epochs - 1:
            print(f'epoch: [{epoch + 1}/{num_epochs}], train_loss: {train_loss:.4f}, '
                  f'valid_loss: {valid_loss:.4f}, '
                  f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}, '
                  f'Optimizer Learning Rate: {optimizer.param_groups[0]["lr"]:.7f}')
        if scheduler is not None:
            # 记录学习率
            LRs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()  # 学习率衰减

        # 存储模型最佳参数
        if best_acc < valid_accs[-1]:
            best_acc = valid_accs[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
    timer.stop()
    # 存储模型
    torch.save(state,
               f'./models/' + model_name + '_best_model' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.pth')
    # 可视化训练效果：静态图
    kk_plot_train_eval_curve(num_epochs, train_losses, train_accs, valid_losses, valid_accs, titles)

    # 打印训练信息
    time_total = timer.sum()
    hours, minutes, seconds = convert_seconds(time_total)
    print(
        f'训练耗时: {hours:.0f}时 {minutes:.0f}分 {seconds:.0f}秒, 平均每秒处理样本数: {(total_trains + total_valids) / timer.stop():.2f} examples/sec, '
        f'最佳精度: {best_acc:.4f}, 训练设备: {str(device)}')

    # 存储训练记录文件
    train_logs = pd.DataFrame({
        'epochs': range(num_epochs),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'valid_loss': valid_losses,
        'valid_acc': valid_accs,
        'best_acc': best_acc,
        'training_time(h:m:s)': str(hours) + ':' + str(minutes) + ':' + str(seconds).split('.')[0],
        'examples/second': (total_trains + total_valids) / timer.stop(),
        'training on device': str(device),
        'Optimizer Learning Rate': LRs[-1]
    })
    train_logs.to_csv(f'./logs/' + model_name + '_train_logs'
                      + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.csv', index=False)
    model.load_state_dict(best_model_wts)
    return model, train_losses, valid_losses, train_accs, valid_accs, LRs


def kk_animator_train_evaluate(model, train_iter, valid_iter, criterion, optimizer, num_epochs,
                               device=None, titles='Examples', plot_epochs=1, scheduler=None,
                               model_name='kkutys'):
    """
    模型训练和评估，可视化展示训练曲线，训练精度和测试精度
    :param model: 模型
    :param train_iter: 训练集数据，一个生成器
    :param valid_iter: 测试集数据，一个生成器
    :param criterion: 损失函数loss
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param num_epochs: 训练轮次
    :param device: 设备类型
    :param titles: 绘图标题
    :param plot_epochs: 绘图轮次设置
    :param model_name: 模型保存名字
    :return: train_loss, train_acc, test_acc
    """
    # 构建一个计时器统计训练时长
    timer = kk_Timer()

    # 构建动态绘图配置
    animator = kk_Animator(xlabel='epoch', ylabel='loss and accuracy', xlim=[1, num_epochs],
                           legend=['train loss', 'train acc', 'valid loss', 'valid acc'], figsize=(8.5, 5.5),
                           title=titles)
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]

    timer.start()  # 开始计时
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc, num_examples_train = kk_train_accuracy(model, train_iter, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        if scheduler is not None:
            # 记录学习率
            LRs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()  # 学习率衰减
        # 评估
        valid_loss, valid_acc, num_examples_test = kk_evaluate_accuracy(model, valid_iter, criterion, device)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # 绘制训练曲线
        if (epoch + 1) % plot_epochs == 0 or epoch == num_epochs - 1:
            animator.add(epoch + 1, (train_loss, train_acc, valid_loss, valid_acc))

    # 存储模型最佳参数
    if best_acc < valid_accs[-1]:
        best_acc = valid_accs[-1]
        best_model_wts = copy.deepcopy(model.state_dict())
    timer.stop()  # 结束计时

    time_total = timer.sum()
    hours, minutes, seconds = convert_seconds(time_total)
    print(
        f'训练耗时: {hours:.0f}时 {minutes:.0f}分 {seconds:.0f}秒, 平均每秒处理样本数: {(num_examples_train + num_examples_test) * num_epochs / time_total:.1f} examples/sec, '
        f'训练loss: {train_loss:.4f}, 测试loss: {valid_loss:.4f} 训练acc: {train_acc:.4f}, 测试acc: {valid_acc:.4f}, '
        f'最佳精度: {best_acc:.4f}, 训练设备: {str(device)}'
        f'最终学习率: {LRs[-1]}')

    return model, train_losses, train_accs, valid_losses, valid_accs, LRs


def kk_evaluate_loss(net, data_iter, loss, device=None):
    """评定给定数据集上模型的损失"""
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    net.to(device)

    metrics = kk_Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            out = net(X)
            l = loss(out, y)
            metrics.add(l.sum(), l.numel())
        return metrics[0] / metrics[1]


def kk_is_files_exist(root_path, prefix):
    """通过文件前缀判断是否在某个目录下存在该文件"""
    search_path = os.path.join(root_path, f'{prefix}_*')
    files = glob.glob(search_path)
    return files

def kk_early_stop(flag, patience, epoch):
    if flag:
        print(f"累计{patience}轮次模型验证精度没有优化，触发早停机制! 当前轮次:  {epoch + 1}")


class kk_ImageClassifierTrainer:
    def __init__(self, config, model, criterion, optimizer, scheduler=None, first_train=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.first_train = first_train
        self.num_epochs = config.num_epochs
        self.patience = config.patience
        self.device = config.device
        self.save_path = config.save_path
        self.logs_path = config.logs_path
        self.plot_titles = config.plot_titles
        self.class_name = config.class_list

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.best_model_wts = None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0.
        self.early_stop = False
        self.batch_list = []
        self.epoch_list = []
        self.batch_counter = 0
        self.update_flag = ''
        self.timer = kk_Timer()
        self.LRs = []

    def train_iter(self, train_loader, val_loader):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        metrics = kk_Accumulator(3)
        if not self.first_train:
            self.best_model_wts = torch.load(self.save_path)
            self.model.load_state_dict(self.best_model_wts)
            self.best_val_acc = pd.read_csv(self.logs_path)['最佳验证精度'].mean().astype(float)
        self.model.to(self.device)
        self.timer.start()
        for epoch in range(self.num_epochs):
            print(f'Epoch: 【{epoch + 1}/{self.num_epochs}】')
            if self.early_stop:
                print(f"累计{self.patience}轮次模型验证精度没有优化，触发早停机制! 当前轮次:  {epoch + 1}")
                break

            # 训练阶段
            self.model.train()
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metrics.add(loss * inputs.shape[0], kk_accuracy(outputs, labels), inputs.shape[0])
                # 每10个batch做一次验证集评估
                if self.batch_counter % 10 == 0:
                    # 学习率
                    self.LRs.append(self.optimizer.param_groups[0]['lr'])
                    epoch_loss = metrics[0] / metrics[2]
                    epoch_acc = metrics[1] / metrics[2]
                    self.train_losses.append(epoch_loss)
                    self.train_accuracies.append(epoch_acc)
                    # 验证阶段
                    val_loss, val_acc = self._evaluate(val_loader)
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_acc)
                    # 早停机制
                    if self.patience is not None:
                        self._early_stopping(val_loss, val_acc)
                    # 打印记录
                    print(f'Iter {self.batch_counter:<6} '
                          f'训练损失: {epoch_loss:<.4f}, '
                          f'训练精度: {epoch_acc:<5.3%}, '
                          f'验证精度: {val_acc:<5.3%}, '
                          f'模型优化: {self.update_flag} '
                          f'训练设备: {str(self.device)}, '
                          f'学习率: {self.LRs[-1]:<.9f}')
                    # 更新学习率
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.batch_list.append(self.batch_counter)
                self.batch_counter += 1

        self.timer.stop()
        time_total = self.timer.sum()
        hours, minutes, seconds = convert_seconds(time_total)
        # 记录日志
        train_logs = pd.DataFrame({
            'Iters': self.batch_list,
            '训练损失': self.train_losses,
            '训练精度': self.train_accuracies,
            '验证损失': self.val_losses,
            '验证精度': self.val_accuracies,
            '最佳验证精度': self.best_val_acc,
            '最小验证损失': self.best_val_loss,
            '训练用时(h:m:s)': str(int(hours)) + ':' + str(int(minutes)) + ':' + str(int(seconds)).split('.')[0],
            '训练设备': str(self.device),
            '学习率': self.LRs
        })
        train_logs.to_csv(os.path.join(self.logs_path, 'train_logs.csv'), index=False)
        print(f"训练轮次: {epoch + 1} 训练耗时: {str(int(hours)) + ':' + str(int(minutes)) + ':' + str(int(seconds)).split('.')[0]} "
              f'训练设备: {str(self.device)}')

    def train_epoch(self, train_loader, val_loader):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        metrics = kk_Accumulator(3)
        if not self.first_train:
            self.best_model_wts = torch.load(self.save_path)
            self.model.load_state_dict(self.best_model_wts)
            self.best_val_acc = pd.read_csv(self.logs_path)['最佳验证精度'].mean().astype(float)
        self.model.to(self.device)
        self.timer.start()
        for epoch in range(self.num_epochs):
            # 学习率
            self.LRs.append(self.optimizer.param_groups[0]['lr'])
            self.epoch_list.append(epoch)
            if self.early_stop:
                print(f"累计{self.patience}轮次模型验证精度没有优化，触发早停机制! 当前轮次:  {epoch + 1}")
                break

            # 训练阶段
            self.model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metrics.add(loss * inputs.shape[0], kk_accuracy(outputs, labels), inputs.shape[0])
            epoch_loss = metrics[0] / metrics[2]
            epoch_acc = metrics[1] / metrics[2]
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            # 验证阶段
            val_loss, val_acc = self._evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            # 早停机制
            if self.patience is not None:
                self._early_stopping(val_loss, val_acc)
            # 打印记录
            print(f'Epoch 【{epoch + 1}/{self.num_epochs}】 '
                  f'训练损失: {epoch_loss:<.4f}, '
                  f'训练精度: {epoch_acc:<5.3%}, '
                  f'验证精度: {val_acc:<5.3%}, '
                  f'模型优化: {self.update_flag} '
                  f'训练设备: {str(self.device)}, '
                  f'学习率: {self.LRs[-1]:.9f}')
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

        self.timer.stop()
        time_total = self.timer.sum()
        hours, minutes, seconds = convert_seconds(time_total)
        # 记录日志
        train_logs = pd.DataFrame({
            'Epochs': self.epoch_list,
            '训练损失': self.train_losses,
            '训练精度': self.train_accuracies,
            '验证损失': self.val_losses,
            '验证精度': self.val_accuracies,
            '最佳验证精度': self.best_val_acc,
            '最小验证损失': self.best_val_loss,
            '训练用时(h:m:s)': str(int(hours)) + ':' + str(int(minutes)) + ':' + str(int(seconds)).split('.')[0],
            '训练设备': str(self.device),
            '学习率': self.LRs
        })
        train_logs.to_csv(os.path.join(self.logs_path, 'train_logs.csv'), index=False)
        print(f"训练轮次: {epoch + 1} 训练耗时: {str(int(hours)) + ':' + str(int(minutes)) + ':' + str(int(seconds)).split('.')[0]} "
              f'训练设备: {str(self.device)}')


    def _evaluate(self, loader):
        self.model.eval()
        metrics = kk_Accumulator(3)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                metrics.add(loss * inputs.shape[0], kk_accuracy(outputs, labels), inputs.shape[0])

        val_loss = metrics[0] / metrics[2]
        val_acc =  metrics[1] / metrics[2]

        return val_loss, val_acc

    def _early_stopping(self, val_loss, val_acc):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        if  val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model_wts = self.model.state_dict().copy()
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
            self.epochs_no_improve = 0
            self.update_flag = '*'
        else:
            self.epochs_no_improve += 1
            self.update_flag = ''

        if self.epochs_no_improve >= self.patience:
            self.early_stop = True

    def plot_training_curves(self, xaixs):
        kk_plot(X=list(xaixs), Y=[self.train_losses, self.train_accuracies, self.val_accuracies],
                xlabel='Iters', ylabel='Loss & Accuracy', xlim=[0, xaixs[-1]],
                legend=['Train Loss', 'Train Accuracy', 'Val Accuracy'],
                titles=[self.plot_titles], figsize=(12, 4))
        plt.show()
        plt.savefig(os.path.join(self.logs_path, 'training_curves.png'))

    def test(self, test_loader):
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        self.model.to('cpu')
        test_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []  # 存储预测概率

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs, dim=1)  # 获取概率
                _, preds = torch.max(outputs, 1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # 计算各项指标
        test_acc = accuracy_score(all_labels, all_preds)
        test_recall = recall_score(all_labels, all_preds, average='macro')
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # 打印测试结果
        print(f'测试损失: {test_loss:<.4f}, 测试精度: {test_acc:<5.3%}, '
              f'测试集 Recall: {test_recall:<5.3%}, 测试集 F1-score: {test_f1:.4f}')
        print('测试集 Confusion Matrix:')
        print(conf_matrix)

        # 创建一个2x2的子图布局
        fig = plt.figure(figsize=(20, 12))

        # 1. 混淆矩阵
        plt.subplot(221)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_name, yticklabels=self.class_name)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # 2. PR曲线
        plt.subplot(222)
        for i in range(len(self.class_name)):
            precision, recall, thresholds = precision_recall_curve(
                (all_labels == i).astype(int), 
                all_probs[:, i]
            )
            plt.plot(recall, precision, label=f'{self.class_name[i]}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. ROC曲线
        plt.subplot(223)
        for i in range(len(self.class_name)):
            fpr, tpr, _ = roc_curve(
                (all_labels == i).astype(int), 
                all_probs[:, i]
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_name[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. F1曲线
        plt.subplot(224)
        for i in range(len(self.class_name)):
            precision, recall, thresholds = precision_recall_curve(
                (all_labels == i).astype(int), 
                all_probs[:, i]
            )
            # 计算每个阈值下的F1分数
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)  # 添加小量防止除零
            plt.plot(thresholds, f1_scores[:-1], label=f'{self.class_name[i]}')  # 注意：f1_scores比thresholds多一个值
            
            # 找到最佳F1分数和对应的阈值
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
            plt.plot(best_threshold, f1_scores[best_f1_idx], 'o', 
                    label=f'{self.class_name[i]} best (t={best_threshold:.2f}, F1={f1_scores[best_f1_idx]:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_path, 'test_metrics.png'), 
                    bbox_inches='tight', dpi=300)
        plt.show()

        # 保存详细的评估指标到CSV文件
        metrics_df = pd.DataFrame({
            'Class': self.class_name,
            'Precision': precision_score(all_labels, all_preds, average=None),
            'Recall': recall_score(all_labels, all_preds, average=None),
            'F1-score': f1_score(all_labels, all_preds, average=None)
        })
        metrics_df.to_csv(os.path.join(self.logs_path, 'test_metrics.csv'), index=False)

    def animator(self, train_loader, val_loader):
        # 构建动态绘图配置
        animator = kk_Animator(xlabel='Epochs', ylabel='Loss and Accuracy', xlim=[1, self.num_epochs],
                               legend=['train loss', 'train acc', 'valid acc'], figsize=(8.5, 5.5),
                               title=self.plot_titles)
        metrics = kk_Accumulator(3)

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                metrics.add(loss * inputs.shape[0], kk_accuracy(outputs, labels), inputs.shape[0])

            train_loss = metrics[0] / metrics[2]
            train_acc = metrics[1] / metrics[2]
            # 评估阶段
            val_loss, val_acc = self._evaluate(val_loader)

            # 动态绘制训练曲线
            if (epoch + 1) % 1 == 0 or epoch == self.num_epochs - 1:
                animator.add(epoch + 1, (train_loss, train_acc, val_acc))