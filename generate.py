from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


def generate_answer(context, question):

    prompt = f"""Medical information:
{context}

Question: {question}

Answer:"""

    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    generated = result[0]["generated_text"]

    answer = generated.split("Answer:", 1)[-1].strip()

    return answer


if __name__ == "__main__":

    context = """
Melanoma is a type of skin cancer that develops from melanocytes,
the cells responsible for producing pigment in the skin.
An image-based AI prediction cannot establish a medical diagnosis.
Suspicious or changing skin lesions should be evaluated by a
qualified healthcare professional.
"""

    question = "What is melanoma?"

    answer = generate_answer(context, question)

    print("\nGenerated Answer:\n")
    print(answer)