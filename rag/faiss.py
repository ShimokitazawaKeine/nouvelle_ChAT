import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.docs.extend(texts)

    def search(self, query_vec, top_k=3):
        D, I = self.index.search(np.array([query_vec]).astype("float32"), top_k)
        return [self.docs[i] for i in I[0]]
