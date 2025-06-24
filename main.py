import requests
import pandas as pd
import time
from prompts import build_prompt
from dotenv import load_dotenv
import os
# LangChain and HuggingFace imports for local pipeline
from transformers import pipeline

# Add Typhoon ChatOpenAI LLM
from langchain_openai import ChatOpenAI

load_dotenv()

# Typhoon v2.1-12b-instruct LLM setup
llm = ChatOpenAI(
    model="typhoon-v2.1-12b-instruct",
    temperature=0,
    max_retries=3,
    base_url="https://api.opentyphoon.ai/v1",
)

def query_mixtral(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return str(response.content).strip()
    except Exception as e:
        print("Error:", e)
        return "ERROR"

import re

def post_process_answer(raw_answer: str) -> str:
    """
    Extracts the first valid keyword from the raw answer.
    Accepts: A, B, C, D, E, Rise, Fall
    """
    match = re.search(r"\b(A|B|C|D|E|Rise|Fall)\b", raw_answer)
    return match.group(1) if match else "INVALID"

def main():
    df = pd.read_csv("data/test.csv")
    results = []
    questions = df.dropna(subset=['query']).to_dict(orient='records')
    for _, row in enumerate(questions, 1):
        prompt = build_prompt(row["query"])
        print("Prompt:", prompt[:100], "...")
        answer = query_mixtral(prompt)
        clean_answer = post_process_answer(answer)
        print("Answer:", clean_answer)
        results.append({"id": row["id"], "answer": clean_answer})
        time.sleep(1.5)  # กัน rate limit

    # Fill unanswered entries with empty strings
    full_ids = df["id"].tolist()
    answered_ids = {entry["id"]: entry["answer"] for entry in results}
    output_data = [{"id": id_, "answer": answered_ids.get(id_, "")} for id_ in full_ids]

    output_df = pd.DataFrame(output_data)
    output_df.to_csv("output/submission.csv", index=False)
    print("✅ Saved to submission.csv")

if __name__ == "__main__":
    main()
