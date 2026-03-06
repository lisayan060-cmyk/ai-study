import gradio as gr
import numpy as np
import cv2
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# single return value exampe
# def reverse_text(text):
#     return text[::-1]

# demo = gr.Interface(
#     fn=reverse_text,
#     inputs="text",
#     outputs="text")

# multiple return values example
# def reverse_and_count(text):
#     reversed_text = text[::-1]
#     count = len(text)
#     return reversed_text, count

# demo = gr.Interface(
#     fn=reverse_and_count,
#     inputs="text",
#     outputs=["text", "number"],
#     title="Reverse and Count",
#     description="Enter a text and get the reversed text and the number of characters in the text.",
#     examples=[
#         "Hello, World!",
#         "Python is fun!",
#         "Gradio is awesome!",
#     ]
# )


# def image_to_sketch(image):
#     gray_image = image.convert('L')
#     inverted_image = 255 - np.array(gray_image)
#     blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
#     inverted_blurred = 255 - blurred
#     pencil_sketch = cv2.divide(np.array(gray_image), inverted_blurred, scale=256.0)
#     return pencil_sketch


# demo = gr.Interface(
#     fn=image_to_sketch,
#     inputs=[gr.Image(label="上传图片", type="pil")],
#     outputs=[gr.Image(label="铅笔画")],
#     title="图像转铅笔画",
#     description="将上传的图片转为铅笔画。"
# )

# demo.launch()

# 从系统环境变量中获取DashScope API密钥
# 确保在运行此脚本前已设置DASHSCOPE_API_KEY环境变量
# api_key = os.getenv("DASHSCOPE_API_KEY")

# # 在模块级别初始化客户端，避免每次调用都重新创建
# client = None
# if api_key:
#     client = OpenAI(
#         api_key=api_key,
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#     )


# def call_qwen(message, history):
#     """
#     调用通义千问max模型的函数

#     参数:
#         message (str): 用户当前输入的消息内容
#         history (list): 聊天历史记录，支持两种格式：
#                         - 格式1: [(用户消息, 助手回复), ...] - 元组列表形式
#                         - 格式2: [{"role": "user", "content": "消息内容"}, ...] - 字典列表形式

#     返回:
#         str: 模型生成的回复内容，如果发生错误则返回格式化的错误信息
#     """
#     if not api_key or not client:
#         return "错误：未设置DASHSCOPE_API_KEY环境变量，请设置后重试。"

#     if not message or not message.strip():
#         return "请输入消息内容。"

#     # 构建消息列表，用于维持对话上下文
#     messages = []

#     # 如果存在历史对话记录，将其添加到消息列表中
#     if history:
#         try:
#             for msg in history:
#                 # 字典格式（较新版本Gradio的格式）
#                 if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
#                     messages.append(msg)
#                 # 元组或列表格式（较旧版本Gradio的格式）
#                 elif isinstance(msg, (list, tuple)) and len(msg) == 2:
#                     user_msg, assistant_msg = msg
#                     if user_msg:
#                         messages.append({"role": "user", "content": str(user_msg)})
#                     if assistant_msg:
#                         messages.append({"role": "assistant", "content": str(assistant_msg)})
#         except Exception as e:
#             print(f"处理历史记录时出错：{e}")

#     messages.append({"role": "user", "content": message})

#     try:
#         response = client.chat.completions.create(
#             model="qwen-max",
#             messages=messages,
#             stream=False
#         )

#         if response.choices and response.choices[0].message.content:
#             return response.choices[0].message.content
#         return "模型未返回有效回复，请重试。"

#     except Exception as e:
#         return "Error: " + str(e)


# # 使用ChatInterface组件，这是Gradio提供的专门用于创建聊天界面的组件
# demo = gr.ChatInterface(
#     fn=call_qwen,
#     title="通义千问-max",
#     description="基于通义千问max的聊天机器人",
#     # 示例问题列表，供用户快速体验
#     examples=[
#         ["你好"],
#         ["你叫什么名字？"],
#         ["给我讲一个笑话噢"]
#     ]
# )

# # 主程序入口点
# # 当直接运行此脚本时，启动Gradio Web服务器
# if __name__ == "__main__":
#     # 启动Gradio服务，默认监听本地7860端口
#     demo.launch(theme=gr.themes.Soft())

# prompt for AI example
load_dotenv()

api_key = os.getenv('DASHSCOPE_API_KEY')
client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def generate_responses(prompt, model="qwen-plus"):
    if not client:
        return "错误：未设置DASHSCOPE_API_KEY环境变量，请在.env文件中设置后重试。"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.7, 
            //near zero means more deterministic, near one means more random
            #max_tokens=128,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return "模型未返回有效回复，请重试。"
    except Exception as e:
        return "Error: " + str(e)


# 1. 定义清楚提示词
# prompt = """
# 根据下面的内容，请答案简短且准确。如果不确定答案，请回答"不确定答案"。

# Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceutical。\
# 在那里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小鼠中提取的。\
# 能够结合到T细胞的表面，并限制它们的细胞杀伤潜力。在1986年，它被批准用于帮助预防肾脏移植后的\
# 器官排斥，成为首个被允许用于人类的治疗性抗体。

# 问题：OKT3最初是从什么来源提取的？
# """
# answer is 小鼠

prompt = """问题：OKT3最初是从什么来源提取的？"""


demo = gr.Interface(
    fn=generate_responses,
    inputs=gr.Textbox(label="输入Prompt", lines=5),
    outputs=gr.Textbox(label="AI回复", lines=10),
    title="Prompt测试工具",
    description="输入Prompt，使用qwen-plus模型生成回复。"
)

if __name__ == "__main__":
    # 测试prompt示例
    response = generate_responses(prompt)
    print(response)

    demo.launch(theme=gr.themes.Soft())
