def build_prompt(question: str) -> str:
    examples = (
        "Question: What is 2 + 2?\n"
        "A: 3\nB: 4\nC: 5\nD: 6\nAnswer: B\n\n"
        "Question: Who is the entrepreneur?\n"
        "A: Barack Obama\nB: James Dyson\nC: Damien Hirst\nD: Mo Farah\nAnswer: B\n\n"
        "Question: Which of these is not a renewable resource?\n"
        "A: Solar\nB: Wind\nC: Coal\nD: Geothermal\nAnswer: C\n\n"
    )
    return (
        "You are a precise multiple-choice and directional question answering assistant.\n"
        "Answer strictly with one of these: A, B, C, D, E, Rise, or Fall. Do not explain your reasoning.\n\n"
        + examples +
        f"Question: {question.strip()}\nAnswer:"
    )