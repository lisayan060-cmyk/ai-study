
#文本向量化
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_embeddings(texts, model="text-embedding-v3"):
    # texts 是一个包含要获取嵌入表示的文本的列表，
    # model 则是用来指定要使用的模型的名称
    # 生成文本的嵌入表示。结果存储在data中。
    data = client.embeddings.create(input=texts, model=model).data
    # print(data)
    # 返回了一个包含所有嵌入表示的列表
    return [x.embedding for x in data]


test_query = ["大模型"]
vec = get_embeddings(test_query)
# "大模型" 文本嵌入表示的列表。
print(vec)
# "大模型" 文本的嵌入表示。
print(vec[0])
# "大模型" 文本的嵌入表示的维度。3072
print(len(vec[0]))
