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

    prompt ="""
[系统规则]
你是检索问答助手。
你必须基于下面的资料作答，并辅佐以自己的知识作补充。
尽可能使用资料中的原句作答。
如果资料不足，回答{"资料不足。"}
最终答案不少于250字。
    
[输出规范]
1. 严禁输出推理过程、思考、链路、草稿、Self-Consistency、CoT 等任何中间内容。
2. 只输出“最终答案”字段。
3. 最多用 1000 汉字；若无法在限长内完整回答，优先给结论，省略解释。
    
[资料]
    """
    for i, ctx in enumerate(contexts):
        prompt += f"[Context {i + 1}]: {ctx.strip()}\n"

    prompt += f"\n[用户问题]\n {question.strip()}\n"

    prompt += "\n[最终答案]\n"

    return prompt


