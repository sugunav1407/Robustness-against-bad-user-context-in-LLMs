import pandas as pd
from reward_collator import RewardDataCollatorWithPadding
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from trl import RewardTrainer
from datasets import Dataset
import torch
from torch import nn

model_name = "argilla/roberta-base-reward-model-falcon-dolly"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

def add_example(ds, accept, reject, tokenizer):
    s1 = tokenizer(accept, return_tensors="pt", truncation=True)
    s2 = tokenizer(reject, return_tensors="pt", truncation=True)
    a, b, c, d = s1["input_ids"][0], s2["input_ids"][0], s1["attention_mask"][0], s2["attention_mask"][0]
    ds["input_ids_chosen"].append(a)
    ds["attention_mask_chosen"].append(c)
    ds["input_ids_rejected"].append(b)
    ds["attention_mask_rejected"].append(d)

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

eval_ds = Dataset.from_dict(eval_ds)
train_ds = Dataset.from_dict(train_ds)

training_args = TrainingArguments(num_train_epochs=25, output_dir="reward_model", per_device_train_batch_size=1, gradient_accumulation_steps=4, remove_unused_columns=False)

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_chosen"])[0]      
        rewards_k = model(input_ids=inputs["input_ids_rejected"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

trainer = RewardTrainer(
args=training_args,
model=model,
train_dataset=train_ds,
eval_dataset=eval_ds,
data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=514),
)

trainer.train()
