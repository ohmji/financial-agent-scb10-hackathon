def build_prompt(question: str) -> str:
    return (
        "Answer the following question with only one of the following choices: A, B, C, D, E, Rise, or Fall.\n"
        "Do not explain your answer. Only respond with the exact single word: A, B, C, D, E, Rise, or Fall.\n\n"
        f"{question.strip()}\n"
        "Answer:"
    )