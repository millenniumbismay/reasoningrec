import torch
import transformers
import json
import os

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score

import random
import numpy as np

from evaluate import load
bertscore = load("bertscore")

import gc

test_output_path = "./lora_llama2_chat/sample128_valsample3000_epoch50_stratified_eval_loss/test_outputs.json"
with open(test_output_path, 'r') as f:
    test_outputs = json.load(f)
print(f"test outputs: {len(test_outputs)}")

references = [test_output['output'] for test_output in test_outputs]
predictions = [test_output['generated_output'] for test_output in test_outputs]
print(f"references: {references[0]} \n\n Predictions: {predictions[0]}")

base_model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(base_model,
                                            padding_side = "left",
                                            add_eos_token = True,
                                            add_bos_token = True,
                                            token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
                                        )
tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})

max_memory_mapping = {0: "23GiB"}
model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory = max_memory_mapping,
            token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
        ).eval()

texts = [references[0], predictions[0]]
t_inputs = tokenizer(texts,
                     return_tensor = "pt",
                     padding="max_length",
                     truncation=True,
                     max_length = 300,
                    )
with torch.no_grad():
    last_hidden_state = model(**t_inputs, output_hidden_states = True).hidden_states[-1]

weights_for_non_padding = t_inputs.attention_mask * torch.arange(start = 1, end = last_hidden_state.shape[1] + 1).unsqueeze(0)
sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

print(sentence_embeddings.shape)