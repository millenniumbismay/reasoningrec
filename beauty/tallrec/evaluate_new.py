import sys

import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
import re
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "baffo32/decapoda-research-llama-7B-hf",
    lora_weights: str = "./lora_llama2_chat/sample_128",
    # lora_weights: str = "/home/grads/m/mbismay/llm-based-seq-rec/ml_1m/ml_1m/tallrec_baseline/lora-llama7b/sample_128",
    test_data_path: str = "./data/test.json",
    result_json_data: str = "llama2_chat_temp_new.json",
    batch_size: int = 32,
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])
    print(f"model_type: {model_type} model_name: {model_name}")

    train_sce = 'fashion'
    test_sce = 'fashion'
    
    # temp_list = model_type.split('_')
    seed = 42
    sample = 128
    
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if not data.__contains__(train_sce):
        data[train_sce] = {}
    if not data[train_sce].__contains__(test_sce):
        data[train_sce][test_sce] = {}
    if not data[train_sce][test_sce].__contains__(model_name):
        data[train_sce][test_sce][model_name] = {}
    if not data[train_sce][test_sce][model_name].__contains__(seed):
        data[train_sce][test_sce][model_name][seed] = {}
    if data[train_sce][test_sce][model_name][seed].__contains__(sample):
        exit(0)
        # data[train_sce][test_sce][model_name][seed][sample] = 

    tokenizer = LlamaTokenizer.from_pretrained(base_model,
                                               token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
                                            )
    max_memory_mapping = {0: "24GiB", 1: "24GiB"}
    print("Loading Model...")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory = max_memory_mapping,
            token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
    # elif device == "mps":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #     )


    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    def evalHelper(inference, cntY, cntN, cntInvalid, invalid_res):
        Yes_No_pattern = r"\b(Yes|No)\b"
        matches = re.findall(Yes_No_pattern, inference, re.IGNORECASE)
        if len(matches) == 0:
            cntInvalid += 1
            invalid_res.append(inference)
            return 0, cntY, cntN, cntInvalid, invalid_res
        for match in matches:
            if match == 'Yes' or match == 'yes':
                cntY += 1
                return 1, cntY, cntN, cntInvalid, invalid_res
            elif match == 'No' or match == 'no':
                cntN += 1
                return 0, cntY, cntN, cntInvalid, invalid_res

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=16,
        batch_size=1,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
            )
        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        # print("output:", output)
        return output, logits.tolist()
        
    # testing code for readme
    logit_list = []
    gold_list= []
    outputs = []
    logits = []
    from tqdm import tqdm
    gold = []
    pred = []

    cntInvalid = cntY = cntN = 0
    invalid_res = []

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['output'] == 'Yes.') for _ in test_data]
        def batch(list, batch_size=32):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output, logit = evaluate(instructions, inputs)
            outputs = outputs + output
            logits = logits + logit
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['logits'] = logits[i]
            # print(f"{outputs[i]} --- {outputs[i][0]} --- {logits[i][0]}")
            # print("--------")
            res, cntY, cntN, cntInvalid, invalid_res = evalHelper(outputs[i], cntY, cntN, cntInvalid, invalid_res)
            pred.append(res)

    print(f"Yes: {cntY} -- No: {cntN} -- Invalid: {cntInvalid}")
    print("Invalid Results: ", invalid_res)

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score

    conf_matrix = confusion_matrix(gold, pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    precision = precision_score(gold, pred)
    print("Precision:", precision)

    recall = recall_score(gold, pred)
    print("Recall:", recall)

    accuracy = accuracy_score(gold, pred)
    print("Accuracy:", accuracy)

    data[train_sce][test_sce][model_name][seed][sample] = roc_auc_score(gold, pred)
    print(data)
    with open(result_json_data, 'w+') as f:
        json.dump(data, f, indent=4)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
