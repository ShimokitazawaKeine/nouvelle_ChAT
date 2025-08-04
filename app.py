from fastapi import FastAPI, Form
from model.qwen_runner import generate_answer
from rag.embedder import embed
from rag.faiss_index import VectorStore

app = FastAPI()

store = VectorStore(dim=384)  # MiniLM 维度是 384

@app.post("/ask")
def ask(question: str = Form(...)):
    query_vec = embed([question])[0]
    top_contexts = store.search(query_vec)

    prompt = "你是一位专业的问答助手。以下是相关信息：\n"
    for i, ctx in enumerate(top_contexts):
        prompt += f"[资料{i+1}]: {ctx}\n"
    prompt += f"\n请根据上述资料回答：{question}"

    answer = generate_answer(prompt)
    return {"answer": answer}
