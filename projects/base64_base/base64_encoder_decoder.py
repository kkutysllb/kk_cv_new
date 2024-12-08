#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-12-08 10:15
# @Desc   : base64编码和解码
# --------------------------------------------------------
"""
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import base64
from PIL import Image

def base64_encode(image_path):
    """base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def base64_decode(encoded_string):
    """base64解码"""
    decoded_image = base64.b64decode(encoded_string)
    return decoded_image

if __name__ == "__main__":
    image_path = "./data/animals/cat.jpg"
    encoded_string = base64_encode(image_path)
    print(encoded_string)
    decoded_image = base64_decode(encoded_string)
    output_path = os.path.join(root_dir, "projects", "base64_base", "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(os.path.join(output_path, "decoded_image.jpg")), "wb") as f:
        f.write(decoded_image)
    
    
