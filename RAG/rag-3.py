# coding=utf-8
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "自然语言处理（NLP），作为计算机科学、人工智能与语言学的交叉之地，致力于赋予计算机解析和处理人类语言的能力。在这个领域，机器学习发挥着至关重要的作用，利用多样的算法，机器得以从海量的文本数据中学习语言的规律和模式。深度学习技术，特别是基于神经网络的模型，如循环神经网络（RNN）和变换器（Transformer），已经在NLP的许多任务中取得了突破性的进展。这些模型能够捕捉语言中的长距离依赖关系，从而在文本分类、情感分析、机器翻译和文本生成等任务上实现了前所未有的性能。随着预训练语言模型（如BERT和GPT系列）的出现，NLP领域迎来了新的革命。这些模型通过在大规模语料库上进行预训练，学习到了丰富的语言表示，能够通过微调适应各种下游任务，极大地推动了NLP技术的发展和应用。"


# ========== 1. 按中文句子结束的标点符号分割 ==========
# 正则表达式匹配中文句子结束的标点符号
sentences = re.split(r'(。|？|！)', text)

# 重新组合句子和结尾的标点符号
chunks = [sentence + (punctuation if punctuation else '') for sentence, punctuation in zip(sentences[::2], sentences[1::2])]

print("=== 按标点符号分割 ===")
for i, chunk in enumerate(chunks):
    print(f"块 {i+1}: {len(chunk)}: {chunk}")

#切割断语义关联
# ========== 2. 按固定长度分割 ==========
def split_by_length(text, chunk_size=100):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


print("\n=== 按固定长度分割 (chunk_size=100) ===")
for i, chunk in enumerate(split_by_length(text, 100)):
    print(f"块 {i+1}: {len(chunk)}: {chunk}")

#提高上下文语义相关性，但是有冗余
# ========== 3. 按重叠窗口分割 ==========
def split_by_overlap(text, chunk_size=100, overlap=20):
    step = chunk_size - overlap
    return [text[i:i+chunk_size] for i in range(0, len(text), step) if i < len(text)]


print("\n=== 按重叠窗口分割 (chunk_size=100, overlap=20) ===")
for i, chunk in enumerate(split_by_overlap(text, 100, 20)):
    print(f"块 {i+1}: {len(chunk)}: {chunk}")


# ========== 4. 递归字符分割 (使用 LangChain RecursiveCharacterTextSplitter) ==========

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

chunks = splitter.split_text(text)

print("\n=== 递归字符分割 (RecursiveCharacterTextSplitter, chunk_size=50, overlap=10) ===")
for i, chunk in enumerate(chunks):
    print(f"块 {i + 1}: {len(chunk)}: {chunk}")
