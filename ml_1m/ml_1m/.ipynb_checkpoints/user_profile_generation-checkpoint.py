from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
import pandas as pd
import numpy as np
import pickle
import tqdm
import torch
print(torch.__version__)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
print(device)

model8bitconfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload = True
)
model_id = "01-ai/Yi-34B-chat"
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = False)

max_memory_mapping = {2: "21GiB", 3: "22GiB", "cpu":"20GiB"}
# max_memory_mapping = {0: "10GiB", 1: "9GiB", 2: "9GiB", 3: "10GiB", "cpu":"20GiB"}
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map = 'auto',
                                             max_memory = max_memory_mapping,
                                             quantization_config = model8bitconfig).eval()

# print("############### Testing #################")
# system_content = """
# You are an expert movie critic. List of user's liked and disliked movies and their descriptions are given in the format -
# Liked Movies: List of movies and their description
# Disliked Movies: List of movies and their description
# Generate a user profile in at most 200 words. Do not include information not present in the movie descriptions.
# """
# content = """
# Liked Movies:
# Back to the Future Part II : Back to the Future Part II takes Marty McFly and Doc Brown on a thrilling time-travel adventure filled with alternate realities, hoverboards, and a race against time.
# Taxi Driver : Taxi Driver (1976) is a gritty and haunting masterpiece directed by Martin Scorsese, exploring the descent into madness of an alienated Vietnam War veteran turned taxi driver.
# Back to the Future : "Back to the Future" is a thrilling and hilarious time-travel adventure where a teenager must ensure his parents' romance to secure his own existence.
# Trainspotting : Trainspotting (1996) is a gritty and intense British film that follows a group of heroin addicts navigating the dark underbelly of Edinburgh.
# Star Wars: Episode V - The Empire Strikes Back : In the epic sequel, the Rebel Alliance faces off against the formidable Empire, while Luke Skywalker confronts his destiny as a Jedi.
# Shawshank Redemption, The : Shawshank Redemption (1994) is a powerful and inspiring drama about friendship, hope, and the resilience of the human spirit within the confines of a prison.
# Magnolia : Magnolia (1999) is a sprawling, emotionally charged drama that weaves together multiple storylines to explore themes of chance, forgiveness, and redemption.
# Dead Man Walking : "Dead Man Walking" (1995) is a powerful and thought-provoking drama that explores the complexities of capital punishment through the eyes of a nun counseling a death row inmate.
# GoodFellas : GoodFellas is a gripping crime drama directed by Martin Scorsese, chronicling the rise and fall of a mobster in 1970s New York City.
# Fargo : "Fargo" is a darkly comedic crime thriller that follows a bumbling car salesman's plan to have his wife kidnapped for ransom, leading to unexpected chaos and violence.
# Insider, The : "The Insider" (1999) is a gripping drama based on true events, exploring the moral dilemma faced by a whistleblower exposing corruption in the tobacco industry.
# E.T. the Extra-Terrestrial : E.T. the Extra-Terrestrial is a heartwarming sci-fi adventure about a young boy who befriends an alien and helps him return home, touching audiences with its magical storytelling.
# Sweet Hereafter, The : The Sweet Hereafter (1997) is a haunting drama that explores the aftermath of a tragic school bus accident and its impact on a small community.
# American History X : American History X is a powerful and thought-provoking drama that explores racism, redemption, and the consequences of hate through the story of a former neo-Nazi.
# Disliked Movies:
# Teenage Mutant Ninja Turtles II: The Secret of the Ooze : Teenage Mutant Ninja Turtles II: The Secret of the Ooze (1991) is a fun-filled sequel with more action, humor, and pizza-loving turtles battling against their nemesis Shredder.
# """

# model = model.eval()
# messages = [
#     {
#         "role": "system",
#         "content": system_content
#     },
#     {
#         "role": "user",
#         "content": content
#      }
# ]
# # messages = [
# #     {
# #         "role": "user",
# #         "content": system_content + content
# #      }
# # ]

# input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# output_ids = model.generate(input_ids.to('cuda'),
#                             # max_new_tokens = 2048,
#                             do_sample = True,
#                             repetition_penalty=1.3,
#                             no_repeat_ngram_size=5,
#                             temperature=0.3,
#                             top_p=0.3
#                             )
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# # Model response: "Hello! How can I assist you today?"
# print(response)

print("#"*100)
print("Creating user profile ...")
print('#'*100)

system_content = """
You are an expert movie critic. List of user's liked and disliked movies and their descriptions are given in the format -
Liked Movies: List of movies and their description
Disliked Movies: List of movies and their description
Generate a user profile in at most 200 words. Do not include information not present in the movie descriptions.
"""

def getUserProfile(model, content):
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": content
        }
    ]
    # print("message:", messages)

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'),
                                do_sample = True,
                                repetition_penalty=1.3,
                                no_repeat_ngram_size=5,
                                temperature=0.3,
                                top_p=0.3
                                )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return response

print(f"Loading user_content_dict...")
with open('user_content_dict.pkl', 'rb') as f:
    user_content_dict = pickle.load(f)
print(len(user_content_dict))
for user, content in user_content_dict.items():
    print(user, content)
    break

print("Loading user_profile_dict...")
with open('user_profile_dict.pkl', 'rb') as f:
    user_profile_dict = pickle.load(f)
print(len(user_profile_dict))
for user, profile in user_profile_dict.items():
    print(user, profile)
    break

with open('user_profile_dict.pkl', 'rb') as f:
    user_profile_dict = pickle.load(f)

model = model.eval()

cnt = 0
for user, content in tqdm.tqdm(user_content_dict.items()):
    # print(user, content)
    cnt += 1
    if cnt <= 4999:
        continue
    user_profile_dict[user] = getUserProfile(model, content)
    # break
    if user%1000 == 0:
        print(user, user_profile_dict[user])
        print("*"*100)
    
    if cnt == 6040:
        break

f = open("user_profile_dict.pkl","wb")
pickle.dump(user_profile_dict,f)
f.close()
