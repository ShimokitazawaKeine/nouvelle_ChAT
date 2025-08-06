# build_index.py

import os
import json
# from tqdm import tqdm

from rag.doc_loader import load_pdf_and_split
from rag.embedder import embed
from rag.faiss_index import VectorStore

DATA_DIR = "data/docs/"
INDEX_FILE = "data/index.faiss"
PASSAGE_FILE = "data/passages.json"

def main():
    all_passages = []

    print(f"[INFO] Loading PDF files from {DATA_DIR}")
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"[INFO] Processing {filename}")
            passages = load_pdf_and_split(filepath)
            all_passages.extend(passages)

    print(f"[INFO] Total passages: {len(all_passages)}")

    print("[INFO] Embedding passages...")
    embeddings = embed(all_passages)

    print("[INFO] Building FAISS index...")
    store = VectorStore(dim=len(embeddings[0]))
    store.add(embeddings, all_passages)
    store.save(INDEX_FILE, PASSAGE_FILE)

    print(f"[DONE] Index saved to: {INDEX_FILE}")
    print(f"[DONE] Passages saved to: {PASSAGE_FILE}")

if __name__ == "__main__":
    main()
