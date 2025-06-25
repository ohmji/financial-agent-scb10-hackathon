def build_prompt(question: str) -> str:
    return (
        "You are a precise multiple-choice and directional question answering assistant. "
        "Answer strictly with one of these: A, B, C, D, E, Rise, or Fall. Do not explain your reasoning.\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )