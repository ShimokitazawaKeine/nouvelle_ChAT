#### simple answering bot based on given documents
For test only, need develop yet

**Before start**: `python build_index.py`
This would build the faiss index for rag search.

**To start:**


backend: you should run `uvicorn app:app --reload` in any terminal under required environment

frontend: run `python -m http.server 8001` in another terminal.


====================施工中===================
目前已完成：本地llm部署、pdf读取与处理、faiss库构建、RAG prompt构建、简单的前端
未完成：openai.api、长对话的prompt构建、树状长对话prompt构建、前端
