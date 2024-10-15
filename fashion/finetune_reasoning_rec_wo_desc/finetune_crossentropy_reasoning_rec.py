import os
import sys
from typing import List
import operator

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, Dataset
import torch.nn.functional as F
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training, ### Only avaialable in peft==0.7.0
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from sklearn.metrics import roc_auc_score
# from bert_score import BERTScorer

# from evaluate import load
# bertscore = load('bertscore')

import gc

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf", #"baffo32/decapoda-research-llama-7B-hf",  # the only required argument
    train_data_path: str = "./final_data/fashion/train.json",
    val_data_path: str = "./final_data/fashion/valid.json",
    output_dir: str = "./lora_llama2_chat/sample128_valsample800_lr1e-4_valauc",
    sample: int = 128,
    val_sample: int = 800,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    cutoff_len: int = 2100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        # "k_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter

):
    print(
        f"Training lora-llama2-chat model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"val_sample: {val_sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created {output_dir}...")
    else:
        print(f"{output_dir} already exists...")

    device_map = 'auto'
    max_memory_mapping = {0: "23GiB", 1: "23GiB"}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print("ddp:", ddp)
    print("Local Rank:", os.environ.get("LOCAL_RANK"))
    print("World Size:", os.environ.get("WORLD_SIZE"))
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or "auto")}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("If ddp: device_map", device_map)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    tokenizer = LlamaTokenizer.from_pretrained(base_model,
                                               padding_side = "left",
                                               add_eos_token = True,
                                               add_bos_token = True,
                                               token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
                                            )

    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})
    # tokenizer.padding_side = "left"  # Allow batched inference
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory = max_memory_mapping,
        token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
    ).eval()
    # model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding='max_length',
            return_tensors=None,
        )
        # if (
        #     result["input_ids"][-1] != tokenizer.eos_token_id
        #     and len(result["input_ids"]) < cutoff_len
        #     and add_eos_token
        # ):
        #     result["input_ids"].append(tokenizer.eos_token_id)
        #     result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        result["text"] = prompt
        # print(result["labels"])

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    

    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    def stratified_sampling(data, sample, seed = 42):
        print("Inside Stratified Sampling helper...")
        # print(data[0])
        data = data.shuffle(seed = seed)
        # print('-'*100)
        # print(f"After suffling: {data[0]}")
        stratified_data = []
        k = 0
        yes_cnt = 0
        no_cnt = 0
        if sample == 800:
            ### a special case
            while no_cnt < sample//2 and k < data.num_rows:
                target =  data[k]['output'].split(' ')[1]
                if target == "No":
                    stratified_data.append(data[k])
                    no_cnt += 1
                k += 1
            k = 0
            while yes_cnt + no_cnt < sample and k < data.num_rows:
                target =  data[k]['output'].split(' ')[1]
                if target == "Yes":
                    stratified_data.append(data[k])
                    yes_cnt += 1
                k += 1
            print(f"Final yes_cnt: {yes_cnt} no_cnt: {no_cnt}")
            return Dataset.from_list(stratified_data)
            

            return Dataset.from_list(stratified_data)
        while yes_cnt < sample//2 or no_cnt < sample//2 and k < data.num_rows:
            # print(k)
            target =  data[k]['output'].split(' ')[1]
            # print(f"Target: {target}")
            if target == 'Yes':
                if yes_cnt < sample//2:
                    stratified_data.append(data[k])
                    yes_cnt += 1
            elif target == 'No':
                if no_cnt < sample//2:
                    stratified_data.append(data[k])
                    no_cnt += 1
            k += 1

        print(f"Final yes_cnt: {yes_cnt} no_cnt: {no_cnt}")
        return Dataset.from_list(stratified_data)
    
    print(train_data)
    train_data["train"] = stratified_sampling(data = train_data["train"], sample = sample, seed = seed)
    # print(train_data)

    # train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    print("Training Data:", train_data)
    # print(train_data["text"])
    # print(train_data[0])
    # print("train_data[0]:", train_data[0])
    # print(tokenizer.batch_decode(train_data[0]['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True))

    print(f"Validation Data: {val_data}")
    val_data["train"] = stratified_sampling(data = val_data["train"], sample = val_sample, seed = seed)
    # val_data["train"] = val_data["train"].shuffle(seed=seed).select(range(val_sample)) if val_sample > -1 else val_data["train"].shuffle(seed=seed)
    val_data["train"] = val_data["train"].shuffle(seed=seed)
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    print("Validation Data:", val_data)
    # print(val_data['text'][:16])
    
    # train_data = train_data.remove_columns(train_data["train"].column_names)
    # print(train_data)

    if not ddp and torch.cuda.device_count() > 1:

        model.is_parallelizable = True
        model.model_parallel = True
    
    def rindex(lst, value):
        return len(lst) - operator.indexOf(reversed(lst), value) - 1

    def compute_metrics(eval_preds):
        gc.collect()
        pre, labels = eval_preds
        pred_labels = pre[0]
        gold = pre[1]
        # print("_"*100)
        # print("pre:", gold, pred_labels)
        # print("Dimensions:", len(labels[0]))
        # print("labels:", labels)
        # print(pre[1], pre[0])
        # print(f"gold: {set(gold)}")
        auc = roc_auc_score(gold, pred_labels)
        print("\nAUC Score:", auc)
        return {'auc': auc}
    
    def cosine_similarity(tensor1, tensor2):
        cosine_similarities = []
        for i in range(len(tensor1)):
            similarity = F.cosine_similarity(tensor1[i].unsqueeze(0), tensor2[i].unsqueeze(0), dim=1)
            cosine_similarities.append(similarity.item())
        print("Cosine Similarities:", cosine_similarities)
        return np.average(cosine_similarities)
    
    # def get_reconstruction_loss(tensor1, tensor2):
    #     cosine_similarities = []
    #     for i in range(len(tensor1)):
    #         similarity = F.cosine_similarity(tensor1[i].unsqueeze(0), tensor2[i].unsqueeze(0), dim=1)
    #         cosine_similarities.append(similarity.item())
    #     print("Cosine Similarities:", cosine_similarities)
    #     return np.average(cosine_similarities)

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        # print("^"*100)
        # print("Inside preprocess_logits_for_metrics")
        # print(f"len of logits: {logits.shape} --- {logits[0]}")
        # print(f"logits dimension: {len(logits[0])}")
        # print(f"len of labels: {labels.shape} --- {labels[0]}")
        # print(f"labels dimension: {len(labels[0])}")
        # gts = []
        # for label in labels.tolist():
        #     temp = []
        #     for num in label:
        #         if num!=-100:
        #             temp.append(num)
        #         else:
        #             temp.append(2)
        #     gts.append(temp)
        # gts = torch.tensor(gts)
        # gts = labels.copy()
        ### Uncomment from here
        # mask_end_idx_list = []
        # for label in labels.tolist():
        #     # print("label:", len(label), label)
        #     mask_end_idx = rindex(label[:-1], -100)
        #     # print("mask_end_idx:", mask_end_idx)
        #     mask_end_idx_list.append(mask_end_idx)
            # print("After removing masks:", label[mask_end_idx+1:-1])
        # print("Mask ends:", len(mask_end_idx_list), mask_end_idx_list)
        # print("labels:", labels)
        # print("GT:", tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=True))
        # print("gts:", gts)
        # gts_text = tokenizer.batch_decode(gts, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        # print("GT:", gts_text)

        # labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        # gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        # labels_index[: , 1] = labels_index[: , 1] - 1

        # logits = logits.softmax(dim=-1)
        # argmax_indices = torch.argmax(logits, dim=-1)
        # print("Predicted Indices:", argmax_indices.shape, argmax_indices)
        
        # preds = []
        # k = 0
        # for predicted in argmax_indices.tolist():
        #     temp = [-100]*mask_end_idx_list[k]
        #     temp.extend(predicted[mask_end_idx_list[k]:])
        #     preds.append(temp)
        #     k += 1
        # preds = torch.tensor(preds)

        # print("-"*100)
        # preds_text = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        # print("Predicted:", preds_text)
        
        # gts = gts.float()
        # preds = preds.float()
        # ### For similarity
        # sim_loss = cosine_similarity(gts.float(), preds.float())
        # print("sim_loss:", sim_loss)

        ### For BERTScore
        # scorer = BERTScorer(model_type = 'microsoft/deberta-large-mnli')
        # bert_score_loss = scorer.score(preds_text, gts_text)
        # print("bert_score_loss:", bert_score_loss)

        ### For Reconstruction Loss
        # reconstruction_loss = get_reconstruction_loss(gts.float(), preds.float())
        # print("reconstruction_loss:", reconstruction_loss)

        ### For AUC
        # print(f"gts:\n{gts[0][-10:]}\n{gts[1][-10:]}")
        # print(f"labels:\n{labels[0][-10:]}\n{labels[1][-10:]}")
        # labels_index = torch.argwhere(torch.bitwise_or(gts == 8241, gts == 3782)) --- From TALLRec
        
        # labels_index = torch.argwhere(torch.bitwise_or(gts == 3869, gts == 1939)) ### Yes - 3869, No - 1939
        # gold = torch.where(gts[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)

        labels_index = torch.argwhere(torch.bitwise_or(labels == 3869, labels == 1939)) ### Yes - 3869, No - 1939
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        # print("labels_index:", labels_index)
        # print("Gold:", gold)
        
        # pred_labels = []
        # logits = logits.softmax(dim=-1)
        # print(f"len of logits: {logits.shape} --- {logits[0]}")
        # for l in labels_index:
        #     yes_prob = logits[l[0]][-3][3869]
        #     no_prob = logits[l[0]][-3][1939]
        #     print("Probability of Yes", logits[l[0]][-3][3869])
        #     print("Probability of No", logits[l[0]][-3][1939])
            # if yes_prob > no_prob:

        # logits = torch.softmax(logits[labels_index[:, 0], -2][:, [3869, 1939]], dim = -1)
        # print(f"Last logits: {logits}")

        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3869, 1939]], dim = -1)
        # print("logits at labels_index", logits, logits[:,0])
        # # print(logits[:, 1][2::3], gold[2::3])
        # print("Formatted logits: ", logits[:,1][2::3].shape, logits[:,1][2::3])
        # print("Formatted labels: ", gold[2::3].shape, gold[2::3])
        
        # return logits[:, 1][2::3], gold[2::3]
        return logits[:,0], gold ### Comment this
        # return logits, gold


    os.environ["WANDB_DISABLED"] = "true"
    
    # if sample > -1:
    #     if sample <= 128 :
    #         eval_step = 25
    #     else:
    #         eval_step = sample / 128 * 16
    eval_step = (sample//batch_size)
    print("eval_step: ", eval_step)
    
    # print("Using Trainer...")
    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         per_device_eval_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=20,
    #         num_train_epochs=num_epochs,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=20,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps",
    #         save_strategy="steps",
    #         eval_steps=eval_step,
    #         save_steps=eval_step,
    #         output_dir=output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_auc",
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to=None,
    #         # report_to="wandb" if use_wandb else None,
    #         # run_name=wandb_run_name if use_wandb else None,
    #         eval_accumulation_steps=1,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    #     # data_collator=transformers.DataCollatorForLanguageModeling(
    #     #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False,
    #     # ),
    #     compute_metrics=compute_metrics,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #     callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    # )

    ### Using SFTTrainer
    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['instruction'])):
    #         text = f"### Instruction: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Response: {example['output'][i]}"
    #         output_texts.append(text)
    #     return output_texts

    instruction_template = "[INST]"
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, instruction_template, tokenizer=tokenizer, mlm=False)

    # print("Using custom TrainerForReconstructionLoss...")    
    print("Using SFTTrainer...")
    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=sample//(batch_size*2),
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=eval_step,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            greater_is_better = True,
            # metric_for_best_model="eval_loss",
            # greater_is_better = False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
            # eval_accumulation_steps=20, ### If this is not set then the entire output from preprocess_logits_for_metrics goes to cpu at last which might lead to CUDA memory issue if logits are not preprocessed
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
        # data_collator=transformers.DataCollatorForLanguageModeling(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False,
        # ),
        max_seq_length = 2100,
        # formatting_func = formatting_prompts_func,
        dataset_text_field="text",
        data_collator = collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=config,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.train()


    model.save_pretrained(output_dir, safe_serialization = False)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    return f"""[INST]{data_point["instruction"]}{data_point["input"]} [/INST]{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
