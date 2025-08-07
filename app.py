from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from rag.embedder import embed
from rag.faiss_index import VectorStore
from model.qwen_runner import generate_answer
from model.template import build_prompt

import numpy as np

# 初始化 FastAPI 实例
app = FastAPI(title="RAG with Qwen")

# 允许跨域请求（如前端使用 React/Streamlit 等调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载向量索引（假设已经 build 好）
store = VectorStore.load("data/index.faiss", "data/passages.json")

@app.post("/ask")
def ask_question(question: str = Form(...)):
    # 向量化问题
    query_vec = embed([question])[0]
    query_vec = np.array(query_vec).astype("float32")

    # 检索最相关的段落（Top-3）
    top_passages = store.search(query_vec, top_k=3)

    # 构造 prompt
    prompt = build_prompt(contexts=top_passages, question=question)

    print(prompt)

    # 调用本地 Qwen 模型生成回答
    answer = generate_answer(prompt)

    print(answer)

    answer = answer["choices"][0]["text"].strip()

    # 返回结果
    return {
        "question": question,
        "prompt": prompt,
        "answer": answer
        # "retrieved": top_passages
    }
