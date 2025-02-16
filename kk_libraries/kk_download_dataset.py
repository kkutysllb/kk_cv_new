#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
# @Author kkutysllb


import sys, os
import hashlib
import tarfile
import zipfile
import requests
import kagglehub

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


'''
下载网上数据集到本地
'''
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'



def kk_download(name, cache_dir=os.path.join('..', 'data')): 
    """下载一个DATA_HUB中的文件, 返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname



def kk_download_extract(name, folder=None): 
    fname = kk_download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir



def kk_download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        kk_download(name)
        



def get_dataset_kaggle_cat_dog():
    # Download latest version
    dataset_path = os.path.join(root_dir, "data", "cat-dog-all")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    path = kagglehub.dataset_download("lizhensheng/cat-dog-all", path=dataset_path)

    print("Path to dataset files:", path)



if __name__ == "__main__":
    get_dataset_kaggle_cat_dog()

