from llama_cpp import Llama

llm = Llama(model_path="model\gguf\Qwen3-0.6B-Q8_0.gguf", n_ctx=2048)

def generate_answer(prompt):
    response = llm(
        prompt=prompt,
        max_tokens=128,
        temperature=0,
        stop=["<|im_end|>"]
    )
    return response["choices"][0]["text"].strip()
