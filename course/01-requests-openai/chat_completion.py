"""使用Python内置库请求OpenAI兼容接口的chat completion示例

这个模块展示如何使用urllib库发送HTTP POST请求到OpenAI兼容的API端点。
核心流程：构造请求数据 -> 发送POST请求 -> 解析响应结果

依赖说明：
    本模块仅使用Python内置包，无需安装任何外部依赖

最小调用方式：
    python chat_completion.py
"""

import json  # 用于JSON数据的序列化和反序列化
import urllib.request  # 用于发送HTTP请求

# 全局常量配置
BASE_URL = "http://143.64.120.39:8695/v1"
API_KEY = "<YOUR_API_KEY>"
MODEL_NAME = "gpt-5-chat"


def main():
    # 构造API端点URL
    url = f"{BASE_URL}/chat/completions"
    
    # 构造请求数据
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个友好的助手"},
            {"role": "user", "content": "say `hello world!`"}
        ]
    }
    
    # 将字典转换为JSON字符串，并编码为字节
    json_data = json.dumps(data).encode("utf-8")
    
    # 构造HTTP请求
    request = urllib.request.Request(
        url,
        data=json_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )
    
    # 发送请求并获取响应
    response = urllib.request.urlopen(request)
    response_data = response.read().decode("utf-8")
    
    # 解析JSON响应
    result = json.loads(response_data)
    
    # 提取并打印AI的回复内容（可以尝试打印result、result["choices"]、result["choices"][0]等不同层次的内容来观察数据结构）
    assistant_message = result["choices"][0]["message"]["content"]
    print(assistant_message)


if __name__ == "__main__":  # 当脚本被直接运行时执行main函数，被导入时不执行
    main()
