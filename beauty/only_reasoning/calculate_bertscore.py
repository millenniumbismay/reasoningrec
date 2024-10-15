import torch
torch.set_num_threads(1)
import transformers
import json
import pickle
import os
import numpy as np
from evaluate import load
bertscore = load("bertscore")

def get_bertscore(predictions, references):
        test_bertscore = bertscore.compute(
                            predictions=predictions,
                            references=references,
                            model_type = "microsoft/deberta-xlarge-mnli")
        return np.average(test_bertscore['f1'])

ground_truth_file = '../finetune_reasoning_rec/final_data/fashion/test.json'
with open(ground_truth_file, 'rb') as f:
    gts = json.load(f)
# print(len(gts), gts[0]['output'])
all_gts = list(gt['output'] for gt in gts)
print(len(all_gts))

pred_file = 'zeroshot_inference_test.json'

with open(pred_file, 'rb') as f:
    predictions = json.load(f)
# print(len(predictions), predictions[1])
all_preds = list(predictions.values())
print(len(all_preds))

print("BERTScore:", get_bertscore(all_preds, all_gts))