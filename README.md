# Robustness against bad user context in LLMs
Conversational agents based on large language models (LLMs) have shown impressive performance in generating accurate responses. However, they often exhibit a lack of contextual robustness, wherein they erroneously change their responses when provided with misleading or incorrect context. In this project, we propose a novel approach to improve the contextual robustness of LLM-based conversational agents using Deep RL based sampling techniques. We implement epsilon-greedy and boltzmann sampling techniques and evaluate the results based on two metrics, robustness and acceptance.

| Experiment | Robustness | Acceptance |
| :---:   | :---: | :---: |
| Baseline | 20.82%   | 70.20%   |
| Epsilon Greedy | 28.98%   | 65.30%   |
| Boltzmann | 35.51%   | 67.34%   |

# Instructions

- eval_ids.txt: The evaluation split from TruthfulQA dataset; numbers correspond to indices in TruthfulQA.csv
  
- experiment_data.csv: The baseline data generated from vanilla dolly-v2-3b model; the first answer generated uses a reference prompt, and is used for other experiments
  
- gen_initial_data.py: Script for generating the baseline (experiment_data.csv) data using the dolly-v2-3b model
  
- ppo/instruct_pipeline.py: Helper script for dolly chat prediction (modified from https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py)
  
- ppo/instruct_pipeline_eval.py: Helper script for dolly chat prediction from saved checkpoint (modified from https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py)

- ppo/eval_script.py: Script used to generate the output from a specific model checkpoint
  
- ppo/ppo_loop_boltzmann.py: PPO loop script for training the model using Boltzmann context sampling
  
- ppo/ppo_loop_random_e.py: PPO loop script for training the model using random epsilon context sampling
  
- ppo/random_sentences.csv: Random sentences dataset
  
- reward_model_training/eval_reward_model.py: Script to evaluate the ranking of the reward model on the evaluation set
  
- reward_model_training/reward_collator.py: Class used to collate reward data from training the model
  
- reward_model_training/train_reward_model.py: Script to train the reward model using ranked training data

