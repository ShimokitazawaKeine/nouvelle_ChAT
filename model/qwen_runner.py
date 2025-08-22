from openai import OpenAI
client = OpenAI(api_key = "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")

def generate_answer(prompt):
    # response = llm(
    #     prompt=prompt,
    #     max_tokens=512,
    #     temperature=0.5,
    #     stop=["<END>"]
    # )


    response = client.responses.create(
        model = "gpt-5-nano-2025-08-07",
        input = prompt,
        # stop = ["<END>"]
    )
    return response

