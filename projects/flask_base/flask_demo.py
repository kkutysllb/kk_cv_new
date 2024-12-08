#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com，31468130@qq.com
# @Date   : 2024-12-08 10:15
# @Desc   : flask demo
# --------------------------------------------------------
"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return "欢饮来到Flask学习世界!"

@app.route("/hello")
def hello():
    return "hello world!"

@app.route("/task")
@app.route("/task/<id>")
def task(id=None):
    print(f"id: {id}")
    return jsonify(
        {   
            'id': id,
            'code': 200,
            'message': 'success',
            'data': [
                    {'img': 'img1',
                 'score': 0.98
                },
                {'img': 'img2',
                 'score': 0.97
                }
            ]
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, debug=True)

