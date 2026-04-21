#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   run_llm_api.py
@Time    :   2026/03/23 14:44:09
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import os
from openai import OpenAI

try:
    _api_key = os.getenv("OPENAI_API_KEY", "")
    _base_url = os.getenv("OPENAI_BASE_URL", "")
    _model = os.getenv("MODEL_NAME", "")

    client = OpenAI(
        # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=_api_key,
        # 各地域的base_url不同
        base_url=_base_url
    )

    completion = client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
    )
    print(completion.choices[0].message.content)
    # 如需查看完整响应，请取消下列注释
    # print(completion.model_dump_json())
except Exception as e:
    print(f"错误信息：{e}")