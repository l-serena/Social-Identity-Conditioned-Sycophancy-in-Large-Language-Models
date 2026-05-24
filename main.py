import json
from datasets import load_dataset

with open("en_single_choice_test_2K.jsonl", encoding="utf-8") as f:
    math_dataset = [json.loads(line) for line in f]

handwriting_dataset = load_dataset("Teklia/IAM-line")
medical_dataset = load_dataset("openlifescienceai/medmcqa") 
finance_dataset = load_dataset("luojunyu/FinMME")


print(math_dataset["train"][0])
print(handwriting_dataset["train"][0])
print(medical_dataset["train"][0])
print(finance_dataset["train"][0])



