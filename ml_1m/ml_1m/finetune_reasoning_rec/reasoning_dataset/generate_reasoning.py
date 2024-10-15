from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
import pandas as pd
import numpy as np
import pickle
import tqdm
import torch
import os
import time
print(torch.__version__)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print("device:", device)

model4bitconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# model8bitconfig = BitsAndBytesConfig(
#     load_in_8bit=True,
#     load_in_8bit_fp32_cpu_offload=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     # llm_int8_has_fp16_weight = False ### Good for finetuning
#     llm_int8_enable_fp32_cpu_offload = True ### To upload to cpu but calculations only happen in the GPU in 8-bit
# )

# model_id = "01-ai/Yi-34B-chat"
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          use_fast = False,
                                          padding_side = "left",
                                          add_eos_token = True,
                                          add_bos_token = True
                                          )
# tokenizer.add_special_tokens({"pad_token":"[PAD]"})
tokenizer.pad_token = tokenizer.eos_token

max_memory_mapping = {0: "23GiB", 1: "23GiB", "cpu":"20GiB"}
# max_memory_mapping = {0: "10GiB", 1: "9GiB", 2: "9GiB", 3: "10GiB", "cpu":"20GiB"}
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map = 'auto',
                                             max_memory = max_memory_mapping,
                                             quantization_config = model4bitconfig).eval()
# model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)

print("#"*100)
print("Generating Reasoning...")
print('#'*100)

def getZeroshotInference(model, content):
    prompts = content

    model_inputs = tokenizer(prompts,
                             padding = True,
                             return_tensors="pt",
                             ).to(device)
    
    outputs = model.generate(**model_inputs,
                             pad_token_id = tokenizer.eos_token_id,
                             max_new_tokens=256,
                             do_sample = True,
                             temperature=0.01,
                             top_p=0.75
                            )
    # print("Outputs: ", outputs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    only_response = [response[i].strip()[len(prompt):] for i, prompt in enumerate(prompts)]
    # print("Response:", response[len(prompt):])
    return only_response

print(f"Loading prompt dataset...")
with open('./reasoning_prompt_data/reasoning_prompt_train_wo_profile.pkl', 'rb') as f:
    prompt_dataset = pickle.load(f)
print(len(prompt_dataset))
for user, content in prompt_dataset.items():
    print(user, content)
    break

target_path = './reasoning_data_wo_profile/reasoning_train_dict.pkl'
if os.path.isfile(target_path):
    print("Loading reasoning_train_dict...")
    with open(target_path, 'rb') as f:
        reasoning_train_dict = pickle.load(f)
    print("Number of users completed:", len(reasoning_train_dict))
    for user, inference in reasoning_train_dict.items():
        print(user, inference)
        break

    # with open('user_profile_dict_mixtral.pkl', 'rb') as f:
    #     user_profile_dict_mixtral = pickle.load(f)
else:
    print("reasoning_train_dict.pkl not found... Creating new dict")
    reasoning_train_dict = dict()

cnt = 0
batch_size = 16
batch_prompts = []
batch_users = []

start = time.time()
batch_start = time.time()
for user, content in tqdm.tqdm(prompt_dataset.items()):
    # print(user, content)
    cnt += 1
    if cnt <= 3200:
        continue
    batch_prompts.append(content[0])
    batch_users.append(user)

    if cnt%batch_size == 0:
        print('_'*100)
        print("Batch Number:", cnt//batch_size)
        # print("Batch prompts: ",len(batch_prompts), batch_prompts)
        # print('-'*100)
        batch_responses = getZeroshotInference(model, batch_prompts)
        # print("Batch Responses: ",len(batch_responses), batch_responses)

        for i in range(len(batch_users)):
            reasoning_train_dict[batch_users[i]] = batch_responses[i] + prompt_dataset[batch_users[i]][1]
        # print("reasoning_train_dict:", len(reasoning_train_dict), reasoning_train_dict)
        batch_prompts = []
        batch_users = []
        print("Time taken for batch:", time.time() - batch_start)
        batch_start = time.time()
    if cnt%(batch_size*5)== 0:
        print(f"Saving at {cnt}...")
        f2 = open(target_path,"wb")
        pickle.dump(reasoning_train_dict,f2)
        f2.close()
    # if user%100 == 0:
    #     print(user, reasoning_train_dict[user])
    #     print("*"*100)
    if cnt == batch_size*300:
        break
print("Time taken for all:", time.time() - start)
f = open(target_path,"wb")
pickle.dump(reasoning_train_dict,f)
f.close()