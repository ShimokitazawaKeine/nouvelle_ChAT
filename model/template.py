# model/template.py

def build_prompt(contexts, question):
    """
    构造一个用于 Qwen 模型的提示词。

    参数：
        contexts: List[str]，从知识库中检索到的段落
        question: str，用户的问题

    返回：
        str，拼接好的 prompt
    """

    prompt = "/nothinkYou are a professional knowledge-based question answering assistant. Please answer the user's question based on the following context.\n\n"

    for i, ctx in enumerate(contexts):
        prompt += f"[Context {i + 1}]: {ctx.strip()}\n"

    prompt += f"\nUser Question: {question.strip()}\n"
    prompt += "Final Answer (based only on the context above):"

    return prompt

