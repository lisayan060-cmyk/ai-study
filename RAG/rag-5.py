# RAG with ChromaDB vector database and Qianwen (DashScope) API
import os
import json
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Purpose: Convert text strings into numerical vectors (embeddings) that capture semantic meaning,
# so similar texts have similar vectors.
def get_embeddings(texts, model="text-embedding-v3", batch_size=10):
    '''封装 DashScope 的 Embedding 模型接口，自动分批处理

    API 返回结构:
        [Embedding(embedding=[0.012, -0.034, ...], index=0), ...]
    '''
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        data = client.embeddings.create(input=batch, model=model).data
        embeddings.extend([x.embedding for x in data])
    return embeddings


def get_completion(prompt, model="qwen-plus"):
    '''封装 DashScope 的聊天模型接口'''
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client.reset()
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, instructions, outputs):
        '''向 collection 中添加文档与向量'''
        embeddings = self.embedding_fn(instructions)
        self.collection.add(
            embeddings=embeddings,
            documents=outputs,
            ids=[f"id{i}" for i in range(len(outputs))],
        )

    def search(self, query, top_n=3):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n,
        )
        return results


if __name__ == "__main__":
    # 读取数据
    with open(os.path.join(os.path.dirname(__file__), 'train_zh.json'), 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    instructions = [entry['instruction'] for entry in data[:100]]
    outputs = [entry['output'] for entry in data[:100]]

    # 创建向量数据库对象
    # Add description for get_embeddings function. 
    #  This function is used to get the embeddings of the text.
    # The get_embeddings function is used to get the embeddings of the text.
    vector_db = MyVectorDBConnector("demo", get_embeddings)

    # 向向量数据库中添加文档
    vector_db.add_documents(instructions, outputs)

    user_query = "营销文案"

    # 1. 检索
    results = vector_db.search(user_query, 2)

    search_results = "\n".join(results['documents'][0])

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
