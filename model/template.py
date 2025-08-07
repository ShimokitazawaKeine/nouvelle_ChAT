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

    prompt = "你是一个知识库问答助手，请根据下面的[Context]作为辅助回答用户的问题。请不要进行思考，请直接输出你的答案。\n\n"

    for i, ctx in enumerate(contexts):
        prompt += f"[Context {i + 1}]: {ctx.strip()}\n"

    prompt += f"\n用户问题: {question.strip()}\n"

    prompt += "当你正式开始回答时，请仅输出一条 'Answer:' 开头的内容。 "

    prompt += "你的回答（以 Answer: 开头）："

    return prompt

