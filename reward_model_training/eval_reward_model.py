import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
from torch import nn
from reward_collator import RewardDataCollatorWithPadding

torch.manual_seed(42)

model_name = "argilla/roberta-base-reward-model-falcon-dolly"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained("reward_model/checkpoint-8000")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def add_example(ds, accept, reject, tokenizer):
    s1 = tokenizer(accept, return_tensors="pt", truncation=True)
    s2 = tokenizer(reject, return_tensors="pt", truncation=True)
    a, b, c, d = s1["input_ids"], s2["input_ids"], s1["attention_mask"], s2["attention_mask"]
    ds["input_ids_chosen"].append(a)
    ds["attention_mask_chosen"].append(c)
    ds["input_ids_rejected"].append(b)
    ds["attention_mask_rejected"].append(d)

train_ds = {"input_ids_chosen": [], "attention_mask_chosen": [], "input_ids_rejected": [], "attention_mask_rejected": []}
eval_ds = {"input_ids_chosen": [], "attention_mask_chosen": [], "input_ids_rejected": [], "attention_mask_rejected": []}

csv_file_path = "reward_train_data.csv"
df = pd.read_csv(csv_file_path)
eval_ids = [int(x) for x in open("eval_ids.txt", "r").read().split("\n")]

for i in range(len(df.index)):
    row = df.loc[i]
    ds = eval_ds if (row["0"] in eval_ids) else train_ds
    q = row["1"]
    fa = row["2"]
    c = row["3"]
    sa = row["4"]

    question = "{}\n\nInput:\nYou said \"{}\". But this is wrong, the right answer is {}".format(q, fa, c)
    passage_format = "Question:\n" + question + "\n\nAnswer:\n"
    passage1 = passage_format + fa + "."
    passage2 = passage_format + sa + "."
    passage3 = passage_format + c + "."

    if i % 2:
        add_example(ds, passage3, passage2, tokenizer)
        add_example(ds, passage2, passage1, tokenizer)

    else:
        add_example(ds, passage1, passage2, tokenizer)
        add_example(ds, passage2, passage3, tokenizer)

tot = 0
passed = 0
inputs = eval_ds

for i in range(len(inputs["input_ids_chosen"])):
    rewards_j = model(input_ids=inputs["input_ids_chosen"][i], attention_mask=inputs["attention_mask_chosen"][i])[0]
    rewards_k = model(input_ids=inputs["input_ids_rejected"][i], attention_mask=inputs["attention_mask_rejected"][i])[0]

    tot += 1
    passed += (1 if rewards_j > rewards_k else 0)

print("Right rankings:", passed, ", total:", tot)