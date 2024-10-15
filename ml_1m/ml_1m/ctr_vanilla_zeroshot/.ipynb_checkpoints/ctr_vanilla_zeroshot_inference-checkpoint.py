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
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = False)

max_memory_mapping = {0: "15GiB", 3: "20GiB", "cpu":"20GiB"}
# max_memory_mapping = {0: "10GiB", 1: "9GiB", 2: "9GiB", 3: "10GiB", "cpu":"20GiB"}
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map = 'auto',
                                             max_memory = max_memory_mapping,
                                             quantization_config = model4bitconfig).eval()

print("#"*100)
print("Generating CTR Vanilla Zeroshot Inference...")
print('#'*100)

def getZeroshotInference(model, content):
    # prompt = """
    #         List of user's liked and disliked movies and their descriptions are given below -
    #         """
    prompt = content
    # prompt += """
    #         You are an expert movie critic. Strictly in 200 words, generate a concise user profile describing the characteristics of movies he likes and dislikes. Do not include information not present in the movie descriptions.
    #         """
    # print("prompt:", prompt)

    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**model_inputs,
                             max_new_tokens=32,
                             do_sample = True,
                             temperature=0.1,
                             top_p=0.75
                            )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print("Response:", response[len(prompt):])
    return response[len(prompt):]

print(f"Loading ctr_vanilla_valid_zeroshot_dataset...")
with open('./ctr_vanilla_zeroshot_dataset/ctr_vanilla_valid_zeroshot_dataset.pkl', 'rb') as f:
    ctr_vanilla_valid_dataset_dict = pickle.load(f)
print(len(ctr_vanilla_valid_dataset_dict))
for user, content in ctr_vanilla_valid_dataset_dict.items():
    print(user, content)
    break

if os.path.isfile('ctr_vanilla_valid_inference_mixtral.pkl'):
    print("Loading ctr_vanilla_valid_inference_dict...")
    with open('ctr_vanilla_valid_inference_mixtral.pkl', 'rb') as f:
        ctr_vanilla_valid_inference_dict = pickle.load(f)
    print("Number of users completed:", len(ctr_vanilla_valid_inference_dict))
    for user, inference in ctr_vanilla_valid_inference_dict.items():
        print(user, inference)
        break

    # with open('user_profile_dict_mixtral.pkl', 'rb') as f:
    #     user_profile_dict_mixtral = pickle.load(f)
else:
    print("ctr_vanilla_valid_inference_mixtral.pkl not found... Creating new dict")
    ctr_vanilla_valid_inference_dict = dict()

cnt = 0
for user, content in tqdm.tqdm(ctr_valid_dataset_dict.items()):
    # print(user, content)
    cnt += 1
    if cnt <= 3599:
        continue
    ctr_vanilla_valid_inference_dict[user] = getZeroshotInference(model, content)
    if cnt%50 == 0:
        print(f"Saving at {cnt}...")
        f2 = open("ctr_vanilla_valid_inference_mixtral.pkl","wb")
        pickle.dump(ctr_vanilla_valid_inference_dict,f2)
        f2.close()
    if user%100 == 0:
        print(user, ctr_vanilla_valid_inference_dict[user])
        print("*"*100)
    if cnt == 6040:
        break

f = open("ctr_vanilla_valid_inference_mixtral.pkl","wb")
pickle.dump(ctr_vanilla_valid_inference_dict,f)
f.close()