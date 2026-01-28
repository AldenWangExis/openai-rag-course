"""使用OpenAI官方SDK请求chat completion示例

第一节的urllib请求确实能实现功能，但是代码比较原始（raw request），
需要手动构造HTTP请求头、处理JSON序列化等底层细节，缺乏语义化的API调用方式。

本节使用OpenAI官方SDK，它提供了更简洁、更具语义化的接口，
自动处理了HTTP通信、认证、错误处理等细节，让代码更易读易维护。

依赖安装：
    pip install openai
    
安装后查看：
    pip list

最小调用方式：
    python chat_completion.py
"""

from openai import OpenAI  # OpenAI官方SDK客户端

# 全局常量配置
BASE_URL = "http://143.64.120.39:8695/v1"
API_KEY = "<YOUR_API_KEY>"
MODEL_NAME = "gpt-5-chat"


def main():
    # 创建OpenAI客户端实例
    # 这里涉及面向对象编程的概念，OpenAI是一个类，client是该类的实例对象，
    # 如果不理解类和对象的概念，建议先学习面向对象程序设计基础教程
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    # 示例1：基础的 hello world
    print("=" * 50)
    print("示例1：基础 hello world")
    print("=" * 50)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个友好的助手"},
            {"role": "user", "content": "say `hello world!`"}
        ]
    )
    print(response.choices[0].message.content)
    print()
    
    # 示例2：通过 system prompt 设定角色，但提示词简单
    print("=" * 50)
    print("示例2：友好助手 + 简单提示词")
    print("=" * 50)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个友好的助手"},
            {"role": "user", "content": "介绍GPT架构"}
        ]
    )
    print(response.choices[0].message.content)
    print()
    
    # 示例3：通过 system prompt 设定专业角色和风格，提示词更具体
    print("=" * 50)
    print("示例3：专业工程师 + 具体提示词")
    print("=" * 50)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是资深LLM工程师，深入浅出，擅长同构类比"},
            {"role": "user", "content": "向大一新生介绍GPT架构"}
        ]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__": 
    main()


"""
===============================================================================
从工程实现到专业理解：LLM 推理的本质
===============================================================================

到这里已经能够通过代码调用 LLM 并获得响应。表面上看，这似乎很简单：
写一个 prompt，发送请求，得到回答。但如果止步于此，就只是"会用工具"。

Prompt 背后：Transformer 的推理过程
-------------------------------------------------------------------------------

向 GPT 模型发送 prompt 时，看起来像是在"许愿"或"下指令"，实际上触发的是
Transformer 架构下的结构化推理过程。这个过程分为两个本质不同的阶段：

1. Prefill 阶段（上下文编码）
   - 模型并行处理整个输入序列
   - 构建注意力机制所需的上下文表示
   - 高度并行的过程，充分利用 GPU 的矩阵运算能力

2. Decoding 阶段（自回归生成）
   - 模型逐个 token 地生成输出序列
   - 每生成一个 token，都基于之前所有 token 的上下文
   - 串行过程，每步依赖前一步的输出

这两个阶段的区别，直接影响推理性能和 prompt 设计策略。

解空间探索：生成的本质是搜索
-------------------------------------------------------------------------------

生成过程本质上是在高维离散空间中的搜索问题：

- 词表大小通常为 50k-100k
- 生成 m 个 token 时，理论解空间大小为 V^m（V 为词表大小）
- 对于一个 100 token 的回答，可能的组合数远超宇宙原子总数

模型通过采样策略在这个巨大的解空间中探索：
- 确定性选择（选择概率最高的 token）
- 随机采样（根据概率分布随机选择）
- 混合策略（temperature、top-p、top-k 等参数控制）

这些参数的调整，本质上是在控制"创造性"和"确定性"之间的平衡。

Prompt Engineering 的本质
-------------------------------------------------------------------------------

理解了推理机制后，prompt engineering 的本质就清晰了：

1. 上下文设计
   - 组织输入信息，使模型能够高效提取关键特征
   - 利用 system prompt 引导模型的输出风格和行为模式

2. 解空间引导
   - 通过示例和指令，约束模型在特定的输出空间中搜索
   - 设计推理链（chain-of-thought），引导模型的生成路径

3. 效率优化
   - 输入长度对计算成本的影响（prefill 的二次复杂度）
   - 控制生成长度以平衡质量和速度

延伸阅读
-------------------------------------------------------------------------------

1. 理论基础
   - 《Attention Is All You Need》：Transformer 原始论文
   - 《Language Models are Few-Shot Learners》：GPT-3 论文
   - Transformer 架构的数学原理和计算复杂度


会调用 API 和理解 API 背后的计算原理，是两个不同的层次。

===============================================================================
"""