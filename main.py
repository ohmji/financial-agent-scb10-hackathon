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

def extract_choice(answer: str) -> str:
    matches = re.findall(r"\b(A|B|C|D|E|Rise|Fall)\b", answer, re.IGNORECASE)
    return matches[-1] if matches else "ERROR"

load_dotenv()


# HuggingFace model setup
hf_model_id = "openthaigpt/openthaigpt-1.6-72b-instruct"

MAX_NEW_TOKENS = 10
SLEEP_TIME = 1.5


tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    torch_dtype="auto", device_map="auto"
)

def query_huggingface(prompt: str) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print("HF Error:", e)
        return "ERROR"






def main():
    df = pd.read_csv("data/test.csv")
    results = []
    questions = df.dropna(subset=['query'])
    for row in questions.itertuples():
        question = row.query
        prompt = build_prompt(question)
        print("Prompt:", prompt[:100], "...")
        raw_answer = query_huggingface(prompt)
        print("Raw answer:", raw_answer)
        clean_answer = extract_choice(raw_answer)
        print("Answer:", clean_answer)
        results.append({"id": row.id, "answer": clean_answer})
        time.sleep(SLEEP_TIME)  # กัน rate limit

    # Fill unanswered entries with empty strings
    full_ids = df["id"].tolist()
    answered_ids = {entry["id"]: entry["answer"] for entry in results}
    output_data = [{"id": id_, "answer": answered_ids.get(id_, "")} for id_ in full_ids]

    output_df = pd.DataFrame(output_data)
    output_df.to_csv("output/submission.csv", index=False)
    print("✅ Saved to submission.csv")

if __name__ == "__main__":
    main()