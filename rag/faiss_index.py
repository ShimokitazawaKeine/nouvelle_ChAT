import faiss
import json
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.passages = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.passages.extend(texts)

    def search(self, query_vec, top_k=3):
        D, I = self.index.search(np.array([query_vec]).astype("float32"), top_k)
        return [self.passages[i] for i in I[0]]

    def save(self, index_path, passages_path):
        faiss.write_index(self.index, index_path)
        with open(passages_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path, passages_path):
        store = cls(dim=384)  # 384 是 MiniLM 的维度
        store.index = faiss.read_index(index_path)
        with open(passages_path, "r", encoding="utf-8") as f:
            store.passages = json.load(f)
        return store
