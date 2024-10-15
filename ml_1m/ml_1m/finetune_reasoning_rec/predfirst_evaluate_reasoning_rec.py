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
from peft import PeftModel, PeftConfig
from scipy.special import softmax
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
import random
import numpy as np

from evaluate import load
bertscore = load("bertscore")

import gc

# from huggingface_hub import login
# access_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
# login(token=access_token)


def main(
    load_8bit: bool = True,
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",
    lora_weights: str = "./lora_llama2_chat/sample64_valsample1500_lr4e-5_wo_profile_valauc",
    # lora_weights: str = "./lora_llama2_chat/sample4096_valsample3000_epoch3_eval_loss",
    test_data_path: str = "./final_data/movie_wo_profile/test.json",
    # result_json_data: str = "./lora_llama2_chat/sample4096_valsample3000_epoch3_eval_loss/results.json",
    # batch_size: int = 32,
    # share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    result_json_data = lora_weights + "/results.json"
    test_output_json_data = lora_weights + "/test_outputs.json"

    model_type = lora_weights.split('/')[-2]
    model_name = lora_weights.split('/')[-1]
    print(f"model_type: {model_type} model_name: {model_name}")

    train_sce = 'movie'
    test_sce = 'movie'
    
    seed = 42
    sample = '64'
    
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
    if not data[train_sce][test_sce][model_name][seed].__contains__(sample):
        data[train_sce][test_sce][model_name][seed][sample] = {}
        # exit(0)
        # data[train_sce][test_sce][model_name][seed][sample] = 


    tokenizer = LlamaTokenizer.from_pretrained(base_model,
                                               padding_side = "left",
                                               add_eos_token = True,
                                               add_bos_token = True,
                                               token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
                                            )
    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})

    max_memory_mapping = {0: "23GiB", 1:"23GiB", 'cpu': "10GiB"}
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

        # peft_config = PeftConfig.from_pretrained(lora_weights)
        # peft_config.init_lora_weights = False
        
        # model.add_adapter(peft_config)
        # model.enable_adapters()

        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0},
            # device_map = "auto",
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


    # tokenizer.padding_side = "left"
    # # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    model = model.bfloat16()
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    def evaluate(
        instructions,
        inputs,
        temperature=0.0,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        batch_size=1,
        max_length=2100,
        **kwargs,
    ):
        # print("-"*100)
        # print("Inside evaluate...")
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        # print("+"*100)
        # print("Prompt:", prompt)
        inputs = tokenizer(prompt,
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length = max_length,
                           ).to(device)
        # print(f"inputs: {inputs.input_ids.shape}")
        generation_config = GenerationConfig(
            # do_sample = True,
            temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                # output_attentions = True,
                # batch_size=batch_size,
            )
        # s = generation_output.sequences
        # print("Generation outputs:", generation_output)
        # scores = generation_output.scores[0].softmax(dim=-1)
        # print("scores:", scores.shape, scores)
        # logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        # logits = torch.tensor(scores[:,[3869, 1939]], dtype=torch.float16).softmax(dim=-1)
        # print(f"logits: {logits}")

        # print(logits.shape)
        # print("__________")
        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        # print(f"s: {s.shape}")
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [res.split('[/INST]')[1][1:] if '[/INST]' in res else res for res in output]
        print("-"*100)
        # print(f"Output: {output}")
        
        generated_s = s[:, max_length:] ### sequences is for the entire prompt + completion part, hence need to take out only the generation part
        # generated_s = s
        # labels_index = torch.argwhere(torch.bitwise_or(generated_s== 3869, generated_s== 1939))
        # labels_index[: , 1] = labels_index[: , 1] - 1
        # print(f"labels_index: {labels_index}")

        ## Check if we are missing any label
        scores = generation_output.scores ### scores are only for generation
        ### Now both scores and generated_s are for generation part
        # print(f"sequences: {generated_s.shape}")
        # print(f"scores: {len(scores)} -- {scores[0].shape}")
        # print(scores[:10])
        logits = []
        # print(f"Checking if we are missing any label")
        k = 0
        for _s in generated_s:
            _s_label_index = torch.argwhere(torch.bitwise_or(_s== 3869, _s== 1939))
            # print(f"k {k}:_s_label_index: {_s_label_index[0]}")
            if len(_s_label_index) < 1:
                # print(f"Couldn't find prediction Yes/No for generation... checking probability of Yes/No from last token")
                score = scores[4].softmax(dim = -1)
                softmax_scores = torch.softmax(score[k, [3869, 1939]], dim = -1)
                # print(f"Score for Yes (3869): {score[k][3869], softmax_scores[0]}")
                # print(f"Score for No (1939): {score[k][1939], softmax_scores[1]}")
                logits.append(softmax_scores[0].item())
                
            else:
                score = scores[_s_label_index[0]].softmax(dim = -1)
                # print(f"score: {score.shape}\n {score}")
                softmax_scores = torch.softmax(score[k, [3869, 1939]], dim = -1)
                # print("Score for Yes (3869):", score[k][3869], softmax_scores[0])
                # print("Score for No (1939)", score[k][1939], softmax_scores[1])
                logits.append(softmax_scores[0].item())

            k += 1
        # print(f"logits: {logits}")

        # scores = generation_output.scores
        # logits = []
        # print(f"len of scores: {len(scores)}")
        # for label in labels_index:
        #     print(f"label: {label}")
        #     score = scores[label[1]].softmax(dim = -1)
        #     print(f"score: {score.shape}\n {score}")
        #     print("Score for Yes (3869):", score[label[0]][3869])
        #     print("Score for No (1939)", score[label[0]][1939])
        #     logits.append([score[label[0]][3869], score[label[0]][1939]])

        
        # logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3869, 1939]], dim = -1)
        # print(f"logits: {logits.shape} \n{logits}")

        # output = [_.split('Response:\n')[-1] for _ in output]
        # print(output, logits.tolist())
        # print("output:", output)
        # return 0
        return output, logits
        
    # testing code for readme
    logit_list = []
    gold_list= []
    outputs = []
    logits = []
    from tqdm import tqdm
    gold = []
    pred = []
    test_sample = 1500
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        # print(type(test_data))
        random.seed(seed)
        test_data = random.sample(test_data, test_sample)
        # test_data = test_data[:test_sample]

        print(f"test data size: {len(test_data)}")
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        golds = [_['output'] for _ in test_data]

        assert all(gold_output.split(' ')[1] in ['Yes','No'] for gold_output in golds), 'Yes/No not present'
        gold_labels = [1 if gold_output.split(' ')[1] == 'Yes' else 0 for gold_output in golds]
        print(f"gold_labels: {len(gold_labels)}")

        # gold = [int(_['output'] == 'Yes.') for _ in test_data]

        def batch(list, batch_size=8):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]

        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs), batch(golds), batch(gold_labels)))):
            gc.collect()
            instructions, inputs, gold, gold_label = batch
            # print(f"gold_label: {gold_label}")
            # print(f"gold: {gold}")
            output, logit = evaluate(instructions, inputs)
            outputs = outputs + output
            logits = logits + logit

        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['generated_output'] = outputs[i]
            test_data[i]['logits'] = logits[i]
        
        with open(test_output_json_data, 'w+') as f:
            json.dump(test_data, f, indent=4)
        


    # print(len(pred), pred[:100])
    # data[train_sce][test_sce][model_name][seed][sample] = roc_auc_score(gold_labels, pred)
    
    ### Calculate BERTScore
    def get_bertscore(predictions, references):
        test_bertscore = bertscore.compute(
                            predictions=predictions,
                            references=references,
                            model_type = "microsoft/deberta-xlarge-mnli")
        return np.average(test_bertscore['f1'])

    data[train_sce][test_sce][model_name][seed][sample]['auc'] = roc_auc_score(gold_labels, logits)
    data[train_sce][test_sce][model_name][seed][sample]['bertscore'] = get_bertscore(outputs, golds)

    print(data)
    with open(result_json_data, 'w+') as f:
        json.dump(data, f, indent=4)
    
    
    # conf_matrix = confusion_matrix(gold, pred)
    # print("Confusion Matrix:")
    # print(conf_matrix)

    # precision = precision_score(gold, pred)
    # print("Precision:", precision)

    # recall = recall_score(gold, pred)
    # print("Recall:", recall)

    # accuracy = accuracy_score(gold, pred)
    # print("Accuracy:", accuracy)
def generate_prompt(instruction, input):
    return f"""[INST]{instruction}{input}[/INST]"""

if __name__ == "__main__":
    fire.Fire(main)
