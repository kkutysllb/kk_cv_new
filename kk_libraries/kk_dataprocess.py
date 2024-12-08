#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb


import torch
import torchvision
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToPILImage
from torchvision.datasets import Food101, DTD, Flowers102, StanfordCars
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import os
from shutil import copy
from .kk_functions import kk_Accumulator, kk_accuracy

"""
自定义数据预处理方法
"""


def kk_split_train_test(data_path, split_ratio=0.1):
    """自定义划分训练集和数据集方法"""

    def mkfile(file):
        if not os.path.exists(file):
            os.makedirs(file)

    # 获取data文件夹下所有文件夹名（即需要分类的类名）
    file_path = data_path
    flower_class = [cla for cla in os.listdir(file_path)]

    # 创建训练集train 文件夹，并由类名在其目录下创建对应子目录
    mkfile(data_path + '/data/train')
    for cla in flower_class:
        mkfile(data_path + '/data/train/' + cla)

    # 创建测试集test文件夹，并由类名在其目录下创建子目录
    mkfile(data_path + '/data/test')
    for cla in flower_class:
        mkfile(data_path + '/data/test/' + cla)

    # 划分比例，训练集 : 测试集 = 9 : 1
    split_rate = split_ratio

    # 遍历所有类别的全部图像并按比例分成训练集和验证集
    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
        images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
        for index, image in enumerate(images):
            # eval_index 中保存验证集val的图像名称
            if image in eval_index:
                image_path = cla_path + image
                new_path = data_path + '/data/test/' + cla
                copy(image_path, new_path)  # 将选中的图像复制到新路径

            # 其余的图像保存在训练集train中
            else:
                image_path = cla_path + image
                new_path = data_path + '/data/train/' + cla
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()
    print("Data Split Processing done!")


def kk_get_data_mean_stdv2(image_dir):
    """统计图像数据集数据的均值、方差和标准差"""

    def get_image_files(image_dir):
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('png', 'jpg', 'jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files

    image_files = get_image_files(image_dir)

    if not image_files:
        raise ValueError("No images found in the specified directory.")

    num_channels = 3  # Assuming RGB images
    mean = np.zeros(num_channels)
    var = np.zeros(num_channels)
    num_pixels = 0

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image_np = np.array(image) / 255.0  # Normalize to [0, 1]

        mean += image_np.mean(axis=(0, 1))
        var += image_np.var(axis=(0, 1))
        num_pixels += 1

    mean /= num_pixels
    var /= num_pixels
    std = np.sqrt(var)
    print(f'Mean: {mean}, Variance: {var}, Standard Deviation: {std}')
    return mean, var, std


def kk_get_data_mean_std(data_path):
    """获取原始数据的均值和方差"""
    # 文件夹路径，包含所有图片文件
    folder_path = data_path

    # 初始化累积变量
    total_pixels = 0
    sum_normalized_pixel_values = np.zeros(3)  # 如果是RGB图像，需要三个通道的均值和方差

    # 遍历文件夹中的图片文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 可根据实际情况添加其他格式
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image_array = np.array(image)

                # 归一化像素值到0-1之间
                normalized_image_array = image_array / 255.0

                # 累积归一化后的像素值和像素数量
                total_pixels += normalized_image_array.size
                sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

    # 计算均值和方差
    mean = sum_normalized_pixel_values / total_pixels

    sum_squared_diff = np.zeros(3)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                image_array = np.array(image)
                # 归一化像素值到0-1之间
                normalized_image_array = image_array / 255.0

                try:
                    diff = (normalized_image_array - mean) ** 2
                    sum_squared_diff += np.sum(diff, axis=(0, 1))
                except:
                    print(f"捕获到自定义异常")
                # diff = (normalized_image_array - mean) ** 2
                # sum_squared_diff += np.sum(diff, axis=(0, 1))

    variance = sum_squared_diff / total_pixels
    std = np.sqrt(variance)
    print(f'mean: {mean}, variance: {variance}, std: {std}')
    return mean, variance, std


def kk_normal(x, mu, sigma):
    """计算正太分布"""
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-0.5 * 1 / sigma ** 2 * (x - mu) ** 2)


def kk_synthetic_data(w, b, num_examples):
    """生成随机数据集"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)

    return X, y.reshape(-1, 1)


def kk_iter_data(batch_size, features, labels):
    """随机打乱数据集每次返回一个批次"""
    idx = np.random.permutation(len(features))

    for i in range(0, len(features), batch_size):
        batch_idx = torch.tensor(idx[i: min(i + batch_size, len(features))])
        yield features[batch_idx], labels[batch_idx]


def kk_load_array(data_arrays, batch_size, is_train=True):
    """调用torch API构造数据批次生成器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# text_labels_fashion_mnist = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
# text_labels_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# text_labels_cifar10 = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def kk_get_labels(labels, text_labels):
    """返回MNIST数据集的文本标签"""
    return [text_labels[int(i)] for i in labels]


def kk_show_gray_images(imgs, rows, cols, titles=None, scale=1.5, mean=(0.5,), std=(0.5,)):
    """可视化图像样本"""
    figsize = (cols * scale, rows * scale)
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    denormalize_mean = [-m / s for m, s in zip(mean, std)]
    denormalize_std = [1 / s for s in std]
    denormalize = transforms.Normalize(mean=denormalize_mean, std=denormalize_std)
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            if img.shape[0] != 1:
                img = img.unsqueeze(0)
            # 图片张量
            denormalized_img = denormalize(img.cpu())
            ax.imshow(denormalized_img.permute(1, 2, 0).numpy())
        else:
            # PIL格式
            ax.imshow(denormalize(img))

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes


def kk_get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def kk_check_folder_exists(path, folder_name):
    """判断某个路径下是否存在某个文件夹或文件"""
    # 合并路径
    folder_path = os.path.join(path, folder_name)

    if os.path.isdir(folder_path):
        print(f'this {folder_name} exists in {path}.')
        return True
    else:
        print(f'this {folder_name} does not exists in {path}')
        return False


def kk_data_transform(flip_prob, train_resize, valid_resize, crop_size, mean, std):
    """定义数据预处理"""
    return {
        'train': transforms.Compose([
            transforms.Resize(train_resize),  # 改变尺寸
            transforms.RandomRotation(45),  # 随机翻转
            transforms.CenterCrop(crop_size),  # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=flip_prob),  # 水平翻转
            transforms.RandomVerticalFlip(p=flip_prob),  # 垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度、对比度、饱和度、色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度率、三通道就是R=G=B
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(valid_resize),  # 改变尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    }


def kk_load_data(dataset_path, batch_size, DataSets, transform, num_works=4):
    """
    加载数据集，并进行预处理，返回生成器
    """
    # 检查是否是特殊数据集
    if DataSets in [Food101, DTD, Flowers102, StanfordCars]:
        # 这些数据集使用 'split' 参数
        data_train = DataSets(root=dataset_path, split='train', transform=transform['train'], download=True)
        data_test = DataSets(root=dataset_path, split='test', transform=transform['valid'], download=True)
    else:
        # CIFAR 等传统数据集使用 'train' 参数
        data_train = DataSets(root=dataset_path, train=True, transform=transform['train'], download=True)
        data_test = DataSets(root=dataset_path, train=False, transform=transform['valid'], download=True)

    # 创建数据加载器
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def kk_loader_train(root_path, batch_size, split_ratio=0.9, resize=None,
                    is_Crop=False, is_Horizontal=False, is_Vertical=False, crop_size=None, crop_padding=None,
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """构建自有数据集的训练集和验证集生成器"""

    # 数据预处理设置
    trans = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if is_Vertical:
        trans.insert(0, transforms.RandomVerticalFlip())  # 垂直翻转
    if is_Horizontal:
        trans.insert(0, transforms.RandomHorizontalFlip())  # 水平翻转
    if is_Crop:
        trans.insert(0, transforms.RandomCrop(size=crop_size, padding=crop_padding))  # 随机裁剪
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 加载数据并预处理
    dataset = ImageFolder(root=root_path, transform=trans)

    # 划分训练集和验证集
    train_data, valid_data = data.random_split(dataset,
                                               (round(split_ratio * len(dataset)),
                                                round((1.0 - split_ratio) * len(dataset))))

    print(f'训练集大小: {len(train_data)}, 验证集大小: {len(valid_data)}')
    # 生成训练集的验证集数据生成器
    train_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_iter = data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_iter, valid_iter


def kk_loader_test(root_path, batch_size, resize=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """构建自有数据集的测试集生成器"""

    # 数据预处理设置
    trans = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 加载数据
    dataset = ImageFolder(root=root_path, transform=trans)
    print(f'测试数据集大小: {len(dataset)}')

    # 生成测试集生成器
    test_iter = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_iter


def kk_imshow(img, mean, std):
    """
    反归一化
    :param img: 图片
    :return:
    """
    denormalize = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
    return denormalize(img)


def kk_show_rgb_images(imgs, rows, cols, titles=None, scale=1.5,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    显示彩色照片/图片
    :param imgs: 展示的图片Image
    :param rows: 展示的行数，也就是展示多少行
    :param cols: 展示的列数，也就是每行展示几个图片
    :param titles: 图片文字标签
    :param scale: 每个图片展示空间的缩放比例，默认是1.5
    :return:
    """
    show = ToPILImage()
    figsize = (cols * scale, rows * scale)
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        plt_img = show(torchvision.utils.make_grid(kk_imshow(img, mean, std))).resize((100, 100))
        ax.imshow(plt_img)
        ax.axis('off')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes


def kk_predict_photo_labels(model, data_iter, text_labels, device=None, n=12, row=3, mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5)):
    """
    测试集图片标签预测
    :param model: 模型
    :param data_iter: 数据集，一个生成器
    :param text_labels: 文字标签列表
    :param device: 设备类型
    :param n: 展示数据大小
    :param row: 展示几行
    :return:
    """
    if isinstance(model, torch.nn.Module):
        if not device:
            device = next(iter(model.parameters())).device
    model.to(device)
    model.eval()
    metric = kk_Accumulator(2)  # 准确数、样本数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            break
        y_hat = model(X)
        metric.add(kk_accuracy(y_hat, y), y.numel())
        preds_labels = kk_get_labels(y_hat.argmax(dim=1), text_labels)
        trues_labels = kk_get_labels(y, text_labels)
        text_labels = [f'T: {true}' + '\n' + f'P: {pred}' for true, pred in zip(trues_labels, preds_labels)]
        kk_show_rgb_images(X[0:n], row, n // row, titles=text_labels, mean=mean, std=std)
    print(f'预测精度: {metric[0] / metric[1] * 100:.2f}%')


def kk_predict_gray_labels(model, data_iter, text_labels, device=None, n=12, row=3, mean=(0.5,), std=(0.5,)):
    """
    测试集图片标签预测
    :param model: 模型
    :param data_iter: 数据集，一个生成器
    :param text_labels: 文字标签列表
    :param device: 设备类型
    :param n: 展示数据大小
    :param row: 展示几行
    :return:
    """
    if isinstance(model, torch.nn.Module):
        if not device:
            device = next(iter(model.parameters())).device
    model.eval()
    model.to(device)
    metric = kk_Accumulator(2)  # 准确数、样本数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            break
        y_hat = model(X)
        metric.add(kk_accuracy(y_hat, y), y.numel())
        preds_labels = kk_get_labels(y_hat.argmax(dim=1), text_labels)
        trues_labels = kk_get_labels(y, text_labels)
        text_labels = [f'T: {true}' + '\n' + f'P: {pred}' for true, pred in zip(trues_labels, preds_labels)]
        kk_show_gray_images(X[0:n].reshape(n, X.shape[-2], X.shape[-1]), row, n // row, titles=text_labels, mean=mean,
                            std=std)
    print(f'预测精度: {metric[0] / metric[1] * 100:.2f}%')


def kk_predict_images_labels(model, data_path, text_labels, device, resize=None, mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)):
    """预测单个图片"""
    if isinstance(model, torch.nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
        model.to(device)

    # 数据预处理
    normalize = transforms.Normalize(mean=mean, std=std)
    if resize:
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize])
    else:
        trans = transforms.Compose([transforms.ToTensor(), normalize])

    for image in os.listdir(data_path):
        # 添加批次维度
        image = Image.open(os.path.join(data_path, image))
        image = trans(image).unsqueeze(0)

        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
        print("预测值：", text_labels[result])
