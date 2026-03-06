import os
from openai import OpenAI
from dotenv import load_dotenv

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
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return "模型未返回有效回复，请重试。"
    except Exception as e:
        return "Error: " + str(e)
