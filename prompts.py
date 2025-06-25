def build_prompt(question: str) -> str:
    return (
        "You are a precise multiple-choice answering assistant.\n"
        "Answer strictly with one of: A, B, C, D, E.\n"
        "Do not explain your answer.\n\n"
        "Example 1:\n"
        "Question: What is 2 + 2?\n"
        "A: 3\nB: 4\nC: 5\nD: 6\nAnswer: B\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )