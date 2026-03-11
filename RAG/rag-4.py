# for matching records based on query questions.
import os
import json
import redis
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 从我们的env文件中加载出对应的环境变量

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 连接 Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 读取数据
with open(os.path.join(os.path.dirname(__file__), 'train_zh.json'), 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 将数据存储到 Redis
for entry in data[:1000]:
    r.set(entry['instruction'], entry['output'])


# 搜索函数，根据关键字搜索 instruction 中包含该关键字的条目
def search_instructions(keyword, top_n=3):
    keys = r.keys(pattern=f"*{keyword}*")
    return [r.get(key) for key in keys[:top_n]]


def get_completion(prompt, model="qwen-plus"):
    '''封装 openai 接口'''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    #user_query = "翻译"
    user_query = "情感分析"

    # 1. 检索
    search_results = "\n".join(search_instructions(user_query, 3))

    # 2. 构建 Prompt
    prompt = f"""
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息：
{search_results}

用户问：
{user_query}

请用中文回答用户问题。
"""
    print("===Prompt===")
    print(prompt)
    print("===Prompt===")

    # 3. 调用 LLM
    response = get_completion(prompt)

    print("===回复===")
    print(response)
