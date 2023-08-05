import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from trl import RewardTrainer
from datasets import Dataset
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
import random
from instruct_pipeline import *
import numpy as np

gen_dataset = open("random_sentences.csv", "r").read().strip().split("\n")

def sample_random(gen_dataset):
    return gen_dataset[random.randrange(0,len(gen_dataset))]

def sample_true_false(row, c_true_false):
    if c_true_false:
        return random.choice(row['Correct Answers'].split(';'))
    else:
        return random.choice(row['Incorrect Answers'].split(';'))

def query(q, context):
    s = "{}\n\nReference:\n{}"
    ans1 = generate_text(s.format(q, context), return_full_text=False)[0]["generated_text"]
    return ans1

def query2(q, prev_answer, c):
    s = "{}\n\nInput:\nYou said \"{}\". But this is wrong, the right answer is {}"
    q1 = s.format(q, prev_answer, c)
    return generate_text(q1, return_full_text=False)[0]["generated_text"]

def sample_query_context(row):
    q = row['Question']
    #print("Question: "+q)
    c_true = random.choice(row['Correct Answers'].split(';'))
    #print("Context true: "+c_true)
    c_false = random.choice(row['Incorrect Answers'].split(';'))
    #print("Context false: "+c_false)
    return [q, c_true, c_false]

#Load checkpoint of reward model
reward_model_name = "argilla/roberta-base-reward-model-falcon-dolly"
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model/checkpoint-15000")
reward_model.eval()
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

csv_file_path = "experiment_data.csv"
df = pd.read_csv(csv_file_path)

csv_truthful_QnA ="TruthfulQA.csv"
TQA =pd.read_csv(csv_truthful_QnA)

# get models
model_name = "databricks/dolly-v2-3b"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
model_ref = create_reference_model(model, num_shared_layers=20)
LLM_tokenizer = AutoTokenizer.from_pretrained(model_name)
model.pretrained_model.save_pretrained("dkjfnvkfj")
# initialize trainer
ppo_config = PPOConfig(
    batch_size=1,
)
# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, LLM_tokenizer)

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=LLM_tokenizer)

epsilon = 0.15

eval_ids = [int(x) for x in open("eval_ids.txt", "r").read().split("\n")]
cnt = 0

np.random.seed(42)
indices = np.random.permutation(len(df.index))

for i in indices:
    row = df.loc[i]
    test = TQA.loc[row['0']]
    if row["0"] in eval_ids: continue
    cnt += 1
    ques = row['1']
    FA = row['2']
    
    if i % 2:
        not_best = False
        best = True

    else:
        not_best = True
        best  = False

    #Sample context 
    no = random.uniform(0,1)
    if no < (epsilon/3):
        context = sample_random(gen_dataset)

    elif (epsilon/3) < no < (2*epsilon/3):
        context = sample_true_false(test, not_best)

    elif no > (2*epsilon/3):
        context = sample_true_false(test, best)
    answer2 = query2(ques,FA,context)
    question = "{}\n\nInput:\nYou said \"{}\". But this is wrong, the right answer is {}".format(ques, FA, context)
    passage_format = "Question:\n" + question + "\n\nAnswer:\n"
    passage2 = passage_format + answer2 + "."
    reward_input = reward_tokenizer(passage2, return_tensors="pt", truncation=True)
    rewards = reward_model(input_ids=reward_input['input_ids'])[0]
    PPO_input_question = LLM_tokenizer(question,return_tensors="pt")["input_ids"][0]
    PPO_input_answer = LLM_tokenizer(answer2)
    while len(PPO_input_answer["input_ids"]) <= 4:
        PPO_input_answer["input_ids"].append(LLM_tokenizer.pad_token_id)
    PPO_input_answer = torch.tensor(PPO_input_answer["input_ids"])
    train_stats = ppo_trainer.step([PPO_input_question], [PPO_input_answer], [rewards[0]])
    if cnt % 10 == 0: model.pretrained_model.save_pretrained(save_directory="ppo_random_e_" + str(cnt))
    if cnt == 105: break

model.pretrained_model.save_pretrained(save_directory="ppo_random_e_final")
