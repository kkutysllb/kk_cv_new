#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-12-08 15:15
# @Desc   : 图像搜索前端接口
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from flask import Flask, request, jsonify
from ImageSearch.src.processer import process_image_search
import base64

app = Flask(__name__)

def base64_encode(image_path):
    """base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def base64_decode(encoded_string):
    """base64解码"""
    decoded_image = base64.b64decode(encoded_string)
    return decoded_image

@app.route("/")
def index():
    return "简易图像搜索系统"


@app.route("/search", methods=["POST"])
def search():
   try:
    #1. 获取用户上传的图片参数
    img = request.form.get("image")
    if img is None:
        return jsonify(
            {
                "code": 400,
                "message": "图片参数错误, 必须给定图像base64转换后的编码作为入参，当前不存在image参数"
            }
        )
    
    #2. 逻辑处理代码(图像恢复， 特征向量提取， 相似图像搜索)
    result = process_image_search(img)
    return jsonify(
        {
        "code": 200,
        "message": "图像搜索成功",
        "data": result
        }
        )
   except Exception as e:
        return jsonify(
            {
                "code": 500,
                "message": "图像搜索失败, 服务器异常，错误信息: {}".format(e)
            }
        )

