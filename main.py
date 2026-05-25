import json
from datasets import load_dataset

with open("en_single_choice_test_2K.jsonl", encoding="utf-8") as f:
    math_dataset = [json.loads(line) for line in f]         # competition math mcqs

handwriting_dataset = load_dataset("Teklia/IAM-line")       # handwriting identification
medical_dataset = load_dataset("openlifescienceai/medmcqa") # medical mcqs, based on mcat
finance_dataset = load_dataset("luojunyu/FinMME")           # financial mcqs

print(math_dataset[0]["problem"])   # "problem", "answer-value", "answer-option-list", "difficulty"

print(medical_dataset["train"][0]["question"])  # "question", "opa", "opb", "opc", "opd", 
                                                #  "cop" (correct option), "choice-type"

print(finance_dataset["train"][0]["question_text"])  # "image", "question_text", "options", "answer"

print(handwriting_dataset["train"][0]["image"])  # "image", "text"









