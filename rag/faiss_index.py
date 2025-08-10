# rag/faiss_index.py  —— 极简稳定版
import faiss
import json
import numpy as np

def _to_f32(a):
    a = np.asarray(a, dtype="float32")
    if a.ndim == 1:
        a = a[None, :]
    return a

class VectorStore:
    """
    与你原始版兼容：
    - save(index_path, passages_path) 仍是两个参数
    - search(query_vec, top_k) 仍然只返回文本列表
    仅增加 metric 选择，默认 'cosine'，会自动归一化并用 IndexFlatIP
    """
    def __init__(self, dim, metric: str = "cosine"):
        assert metric in {"cosine", "l2"}
        self.dim = dim
        self.metric = metric
        self.use_cosine = (metric == "cosine")
        self.index = faiss.IndexFlatIP(dim) if self.use_cosine else faiss.IndexFlatL2(dim)
        self.passages = []

    def _maybe_norm(self, X):
        X = _to_f32(X)
        if self.use_cosine:
            faiss.normalize_L2(X)  # 就地归一化
        return X

    def add(self, embeddings, texts):
        X = self._maybe_norm(embeddings)
        if X.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {X.shape[1]} != store.dim {self.dim}")
        self.index.add(X)
        self.passages.extend(texts)

    def search(self, query_vec, top_k=3):
        q = self._maybe_norm(query_vec)
        D, I = self.index.search(q, top_k)
        return [self.passages[i] for i in I[0]]

    def save(self, index_path, passages_path):
        faiss.write_index(self.index, index_path)
        with open(passages_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path, passages_path, dim, metric: str = "cosine"):
        store = cls(dim=dim, metric=metric)
        store.index = faiss.read_index(index_path)
        with open(passages_path, "r", encoding="utf-8") as f:
            store.passages = json.load(f)
        return store
