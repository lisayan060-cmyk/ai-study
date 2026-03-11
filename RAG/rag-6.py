"""
RAG Bot 示例

1. 加载 train_zh.json 作为知识库，embedding 后存入向量数据库
2. 用户输入文本 embedding，通过相似度查询从向量数据库中检索相关片段
3. 将检索到的上下文填充到 Prompt 模板，交给 LLM 生成回答
"""

import os
import json
import importlib

rag_5 = importlib.import_module("rag-5")
MyVectorDBConnector = rag_5.MyVectorDBConnector
get_embeddings = rag_5.get_embeddings
get_completion = rag_5.get_completion

prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息：
__INFO__

用户问：
__QUERY__

请用中文回答用户问题。
"""

#help generate kwargs example:
# kwargs = {
#     "info": ["doc1 text", "doc2 text"],
#     "query": "math question",
# }

def build_prompt(prompt_template, **kwargs):
    '''将 Prompt 模板赋值'''
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v) 
        else:
            val = str(v)
        # perform replace operation for each key in kwargs
        # 已知信息：
        # doc1 text
        # doc2 text
        # 用户问：
        # 拉肚子怎么办？
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt


class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, info=search_results['documents'][0], query=user_query)

        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response


if __name__ == "__main__":
    # 读取数据
    with open(os.path.join(os.path.dirname(__file__), 'train_zh.json'), 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    instructions = [entry['instruction'] for entry in data[:1000]]
    outputs = [entry['output'] for entry in data[:1000]]

    # 创建向量数据库对象
    vector_db = MyVectorDBConnector("demo", get_embeddings)

    # 向向量数据库中添加文档
    vector_db.add_documents(instructions, outputs)

    # 创建一个RAG机器人
    bot = RAG_Bot(
        vector_db,
        llm_api=get_completion
    )

    user_query = "数学"

    response = bot.chat(user_query)

    print(response)
