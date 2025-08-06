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

    prompt = "你是一位专业的知识问答助手，请根据以下资料回答用户的问题。\n\n"

    for i, ctx in enumerate(contexts):
        prompt += f"[资料{i+1}]: {ctx.strip()}\n"

    prompt += f"\n用户问题：{question.strip()}\n"
    prompt += "请结合资料，用简洁清晰的语言回答："

    return prompt
