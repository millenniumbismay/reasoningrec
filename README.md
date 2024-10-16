# ReasoningRec

## Abstract
This paper presents ReasoningRec, a reasoning-based recommendation framework that leverages Large Language Models (LLMs) to bridge the gap between recommendations and human-interpretable explanations. In contrast to conventional recommendation systems that rely on implicit user-item interactions, ReasoningRec employs LLMs to model users and items, focusing on preferences, aversions, and explanatory reasoning. The framework utilizes a larger LLM to generate synthetic explanations for user preferences, subsequently used to fine-tune a smaller LLM for enhanced recommendation accuracy and human-interpretable explanation. Our experimental study investigates the impact of reasoning and contextual information on personalized recommendations, revealing that the quality of contextual and personalized data significantly influences the LLM's capacity to generate plausible explanations. Empirical evaluations demonstrate that ReasoningRec surpasses state-of-the-art methods by up to 12.5\% in recommendation prediction while concurrently providing human-intelligible explanations.

## Process to Run Code
Every dataset has its own folder. Due to constraints on size limite, only code files are provided. Input data, output data, and model files are not provided in the zip. They would be available on the public repo after review. Please follow the steps below to reproduce the results. All process are described wrt Fashion dataset - `reasoningrec/fashion/` and would be very similar for Beauty and ML1M datasets.
### Datasets
ML1M - https://grouplens.org/datasets/movielens/1m/
Amazon - https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/#subsets

### Data Preparation
All preparation and preprocessing steps is present in `reasoningrec/fashion/fashion_data/`. The `processed_data` folder contains all the data required for our experiments.
`create_fashion_data.ipynb` - cleans the raw data and creates the initial version of data
`create_item_description.py` - generates item description of 25-words each
`create_user_profile_summary.py` - generates user profile of 100-words each

### Zero-shot Methods
Assume that we are in the `reasoningrec/fashion/ctr_zeroshot_2` folder
Folders with `zeroshot` in their name.
Use the `create_ctr_zeroshot_data.ipynb` or similarly named notebook to create the prompt data - It will be stored in `/ctr_zeroshot_2/ctr_zeroshot_dataset`
Use `zeroshot_inferencel.py` to run the Mixtral model on these prompts -- The output would be stored in the current folder as `zeroshot_inference.py`
Use `evaluate.ipynb` to evaluate the result to get the AUC scores
Use `calculate_bertscore.py` to evaluate the BERTScore result for methods generating reasoning. Might need to be tweaked based on the ground truth reasoning data.

### Finetuned Methods
Assume that we are in the `reasoningrec/fashion/finetune_reasoning_rec_2` folder -- This is a representative and all folders are structured very similarly
#### Generate Reasoning Ground Truth
`reasoningrec/fashion/finetune_reasoning_rec_2/reasoning_dataset` folder has the generated reasoning ground truth data
Use `create_reasoning_prompts.ipynb` to generate prompts for creating reasoning ground truths and store them in `./reasoning_prompt_data/` folder
Use `generate_reasoning*.py` to generate groud truth reasoning and store them in `./reasoning_data/` folder
#### Generate Finetuning Data
Use `create_final_data.ipynb` to create the finetuning data in the format - {'instruction': '', 'input': '', 'output': ''}
The train, test, and validation data would be stored in `./final_data/` folder
#### Instruction Finetuning and Evaluation
Create a folder `./lora_llama2_chat/`
Use `finetune_crossentropy_reasoning_rec.py` to finetune the model. Appropriately set the parameters like K from {64, 128, 254}
Use GPU as required by setting the max_memory_mapping
The LoRA model would be stored in the `./lora_llama2_chat/` folder with metrics details

Once the model is trained,
Use `predfirst_evaluate_reasoning_rec.py` to evaluate the model -- Use the name of the model properly
This would create a file `./lora_llama2_chat/model_name/test_outputs.json` which contains the generated outputs with logits in the following format - {'instruction': '', 'input': '', 'output': '', 'generated_outputs': '', 'logits': float()}
