import requests
import pandas as pd
import time
from prompts import build_prompt
from dotenv import load_dotenv
import os
# LangChain and HuggingFace imports for local pipeline
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

load_dotenv()


# HuggingFace model setup
hf_model_id = "aisingapore/Gemma2-9b-WangchanLIONv2-instruct"


tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    device_map="auto",
)

def query_huggingface(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10,temperature=0)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print("HF Error:", e)
        return "ERROR"





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
        answer = query_huggingface(prompt)
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