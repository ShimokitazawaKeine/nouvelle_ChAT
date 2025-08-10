from llama_cpp import Llama

llm = Llama(model_path="model\gguf\Qwen3-4B-Q5_K_M.gguf", n_ctx=2048)

def generate_answer(prompt):
    response = llm(
        prompt=prompt,
        max_tokens=512,
        temperature=0.5,

        # repeat_penalty=1.1,
        # logit_bias={int(nl_id): -2.0},

        stop=["<END>"]
    )
    # return response["choices"][0]["text"].strip()
    return response