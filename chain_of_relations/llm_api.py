#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   llm_api.py
@Time    :   2026/02/11 08:35:53
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import os
import time
import logging
from openai import OpenAI


class LLMAPI(object):

    def __init__(self, model_name=None, model_type=None):
        self.model_name = model_name or model_type or os.getenv("MODEL_NAME")
        
        # 从环境变量读取配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.retry_interval = float(os.getenv("OPENAI_RETRY_INTERVAL", "1"))

        if not self.model_name:
            raise ValueError("MODEL_NAME not found in environment variables and no model_name/model_type provided")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # 统一初始化OpenAI客户端（支持所有兼容OpenAI接口的服务商）
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logging.info(
                f"Initialized OpenAI client with model: {self.model_name}, "
                f"base_url: {self.base_url}, timeout: {self.timeout}s"
            )
        else:
            self.client = OpenAI(api_key=self.api_key)
            logging.info(
                f"Initialized OpenAI client with model: {self.model_name}, "
                f"timeout: {self.timeout}s"
            )

    def generate(self, user_prompt, temperature=0.01, max_tokens=256, system_prompt=None):
        """
        统一使用OpenAI标准接口进行调用
        支持所有兼容OpenAI API的服务商：OpenAI、Pumpkin、SiliconFlow、DeepSeek等
        """
        logging.info(f"Requesting model: {self.model_name}")

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        last_error = None
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
            "timeout": self.timeout,
        }
        if str(self.model_name).startswith("gpt-5"):
            request_kwargs["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "low")

        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request_kwargs)

                result = None
                if getattr(response, "choices", None):
                    message = getattr(response.choices[0], "message", None)
                    if message is not None:
                        result = getattr(message, "content", None)

                usage_obj = getattr(response, "usage", None)
                if usage_obj:
                    usage = {
                        "input_tokens": getattr(usage_obj, "prompt_tokens", 0),
                        "output_tokens": getattr(usage_obj, "completion_tokens", 0),
                        "total_tokens": getattr(usage_obj, "total_tokens", 0),
                    }

                if result:
                    return result, usage

                last_error = RuntimeError("empty_model_response")
                logging.warning(
                    f"Model response has empty content, retry {attempt}/{self.max_retries}"
                )
            except Exception as e:
                last_error = e
                logging.error(
                    f"API error ({e}), retry {attempt}/{self.max_retries}"
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_interval)

        logging.error(f"Failed to get response after {self.max_retries} retries")
        usage["error"] = str(last_error) if last_error else "unknown_error"
        return None, usage
