from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import numpy as np
import pickle
import tqdm
import torch
import os
import time
print(torch.__version__)
import json

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
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "meta-llama/Llama-2-7b-chat-hf"
print(f"Loading {model_id}...")
tokenizer = LlamaTokenizer.from_pretrained(model_id,
                                               padding_side = "left",
                                               add_eos_token = True,
                                               add_bos_token = True,
                                               token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
                                            )

tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})
max_memory_mapping = {0: "23GiB", 1: "23GiB", "cpu":"20GiB"}
# max_memory_mapping = {0: "10GiB", 1: "9GiB", 2: "9GiB", 3: "10GiB", "cpu":"20GiB"}
model = LlamaForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto',
        max_memory = max_memory_mapping,
        token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
    ).eval()

print("#"*100)
print("Generating CTR Vanilla Zeroshot Inference...")
print('#'*100)

def getZeroshotInference(model, content):
    prompts = content

    model_inputs = tokenizer(prompts,
                             padding = True,
                             return_tensors="pt",
                             ).to(device)
    
    outputs = model.generate(**model_inputs,
                             pad_token_id = tokenizer.eos_token_id,
                             max_new_tokens=300,
                             do_sample = True,
                             temperature=0.01,
                             top_p=0.9
                            )
    # print("Outputs: ", outputs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    only_response = [response[i].strip()[len(prompt):] for i, prompt in enumerate(prompts)]
    # print("Response:", response[len(prompt):])
    return only_response

print(f"Loading ctr_vanilla_test_zeroshot_dataset...")
with open('./ctr_vanilla_zeroshot_dataset/ctr_vanilla_test_zeroshot_dataset.pkl', 'rb') as f:
    ctr_vanilla_test_dataset_dict = pickle.load(f)
print(len(ctr_vanilla_test_dataset_dict))
for user, content in ctr_vanilla_test_dataset_dict.items():
    print(user, content)
    break

if os.path.isfile('ctr_vanilla_test_inference_llama.json'):
    print("Loading ctr_vanilla_test_inference_dict...")
    with open('ctr_vanilla_test_inference_llama.json', 'rb') as f:
        ctr_vanilla_test_inference_dict = json.load(f)
    print("Number of users completed:", len(ctr_vanilla_test_inference_dict))
    for user, inference in ctr_vanilla_test_inference_dict.items():
        print(user, inference)
        break

    # with open('user_profile_dict_mixtral.pkl', 'rb') as f:
    #     user_profile_dict_mixtral = pickle.load(f)
else:
    print("ctr_vanilla_test_inference_llama.json not found... Creating new dict")
    ctr_vanilla_test_inference_dict = dict()

# cnt = 0
# for user, content in tqdm.tqdm(ctr_vanilla_test_dataset_dict.items()):
#     # print(user, content)
#     cnt += 1
#     # if cnt <= 999:
#     #     continue
#     ctr_vanilla_test_inference_dict[user] = getZeroshotInference(model, content)
#     if cnt%50 == 0:
#         print(f"Saving at {cnt}...")
#         f2 = open("ctr_vanilla_test_inference_llama.pkl","wb")
#         pickle.dump(ctr_vanilla_test_inference_dict,f2)
#         f2.close()
#     if user%100 == 0:
#         print(user, ctr_vanilla_test_inference_dict[user])
#         print("*"*100)
    # if cnt == 6040:
    #     break

# f = open("ctr_vanilla_test_inference_mixtral.pkl","wb")
# pickle.dump(ctr_vanilla_test_inference_dict,f)
# f.close()

cnt = 0
batch_size = 20
batch_prompts = []
batch_users = []
batch_prompts_cnt = 0

import gc
gc.collect()
torch.cuda.empty_cache()
start = time.time()
batch_start = time.time()

for user, prompt in tqdm.tqdm(ctr_vanilla_test_dataset_dict.items()):
    # print(user, prompt)
    cnt += 1
    # if cnt <= 4800:
    #     continue
    if user in ctr_vanilla_test_inference_dict:
        continue
    
    batch_prompts.append(prompt)
    batch_users.append(user)
    batch_prompts_cnt += 1

    if batch_prompts_cnt == batch_size:
        print('_'*100)
        print("Batch Number:", cnt//batch_size)
        print("Batch prompts: ",len(batch_prompts))
        batch_prompts_length = sum([len(review.split(' ')) for review in batch_prompts])
        print(f"Length of batches: {batch_prompts_length}")
        
        batch_responses = getZeroshotInference(model, batch_prompts)
        # print("Batch Responses: ",len(batch_responses), batch_responses)

        for i in range(len(batch_users)):
            ctr_vanilla_test_inference_dict[str(batch_users[i])] = batch_responses[i]
        # print("reasoning_train_dict:", len(reasoning_train_dict), reasoning_train_dict)
        batch_prompts = []
        batch_users = []
        batch_prompts_cnt = 0
        gc.collect()
        torch.cuda.empty_cache()
        print("Time taken for batch:", time.time() - batch_start)
        batch_start = time.time()
    if cnt%(batch_size*2)== 0:
        print(f"Saving at {cnt}...")
        with open('ctr_vanilla_test_inference_llama.json',"w+") as f:
            json.dump(ctr_vanilla_test_inference_dict,f)
    gc.collect()
    torch.cuda.empty_cache()

    # if cnt == batch_size*2:
    #     break

print("Time taken for all:", time.time() - start)
with open('ctr_vanilla_test_inference_llama.json',"w+") as f:
    json.dump(ctr_vanilla_test_inference_dict,f)