def build_prompt(question: str) -> str:
    return (
        "Answer the following question with only one of the following choices: A, B, C, D, E, Rise, or Fall.\n"
        "Do not explain your answer.\n\n"
        f"{question.strip()}\n"
        "Answer:"
    )