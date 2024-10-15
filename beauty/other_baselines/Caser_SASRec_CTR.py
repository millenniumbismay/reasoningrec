import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SASRecModules_ori import *
import random
import json
import copy
from sklearn.metrics import roc_auc_score



logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='beauty',
                        help='book, movie, fashion, beauty')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=16,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=4,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-5,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument("--model", type=str, default="Caser",
                        help='the model name, GRU, Caser, SASRec')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument("--early_stop", type=int, default=20,
                        help='the epoch for early stop')
    parser.add_argument("--eval_num", type=int, default=1,
                        help='evaluate every eval_num epoch' )
    parser.add_argument("--seed", type=int, default=42,
                        help="the random seed")
    parser.add_argument("--randon_sample_num", type=int, default=256,
                        help="the random seed")
    parser.add_argument("--result_json_path", type=str, default="./result_temp/temp.json")
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        # indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        # state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

class Caser_with_label_CTR(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes, dropout_rate):
        super(Caser_with_label_CTR, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def get_mask(self, len_states, max_length):
        len_states = len_states.cuda()
        n_sequences = len_states.size(0)
        index_tensor = torch.arange(max_length).unsqueeze(0).repeat(n_sequences, 1).cuda()
        mask = (index_tensor < len_states.unsqueeze(1)).int().unsqueeze(-1)
        return mask

    def forward(self, states, state_rate, len_states):
        input_emb = self.item_embeddings(states)
        state_rate_reshape = state_rate.view(-1,5,1)
        input_emb = torch.cat((input_emb, state_rate_reshape), dim=2)
        # input_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        mask = self.get_mask(len_states, states.shape[1])

        # mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        # print(f"horizontal_cnn: {self.horizontal_cnn}")
        # print(f"input_emb shape: {input_emb.shape}")
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            # print(f"h_out shape: {h_out.shape}")
            h_out = h_out.squeeze()
            # print(f"h_out shape: {h_out.shape}")
            h_out_reshaped = h_out.view(-1, h_out.shape[1], h_out.shape[2])
            p_out = nn.functional.max_pool1d(h_out_reshaped, h_out.shape[2])
            # print(f"p_out shape: {p_out.shape}")
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)
        # print(f"h_pool_flat shape: {h_pool_flat.shape}")

        # print(f"Vertical cnn: {self.vertical_cnn}")
        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        # print(f"v_out shape: {v_out.shape}")
        v_out = v_out.squeeze()
        v_out = torch.cat([v_out, v_out], dim=0)
        # print(f"v_out shape: {v_out.shape}")
        # v_flat = v_out.view(-1, self.hidden_size)
        # v_flat = v_out.reshape(v_out.shape[0]*2, self.hidden_size)
        v_flat = v_out[:, :self.hidden_size]
        # print(f"v_flat shape: {v_flat.shape}")

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        # input_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        mask = self.get_mask(len_states, states.shape[1])
        # mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            h_out_reshaped = h_out.view(-1, h_out.shape[1], h_out.shape[2])
            p_out = nn.functional.max_pool1d(h_out_reshaped, h_out.shape[2])
            # print(f"p_out shape: {p_out.shape}")
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_out = v_out.squeeze()
        v_out = torch.cat([v_out, v_out], dim=0)
        # v_flat = v_out.view(-1, self.hidden_size)
        v_flat = v_out[:, :self.hidden_size]

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        
        return supervised_output

class SASRec_with_label_CTR(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        # print(hidden_size, item_num, state_size, dropout, device)
        super(SASRec_with_label_CTR, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        # embedding dim minors one for rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num,
            embedding_dim=hidden_size - 1,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()
    
    def get_mask(self, len_states, max_length):
        len_states = len_states.cuda()
        n_sequences = len_states.size(0)
        index_tensor = torch.arange(max_length).unsqueeze(0).repeat(n_sequences, 1).cuda()
        mask = (index_tensor < len_states.unsqueeze(1)).int().unsqueeze(-1)
        return mask


    def forward(self, states, state_rate, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        state_rate_reshape = state_rate.view(-1,5,1)
        # print(len(inputs_emb), state_rate_reshape)
        inputs_emb = torch.cat((inputs_emb, state_rate_reshape), dim=2)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = self.get_mask(len_states, states.shape[1])
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, state_rate, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        state_rate_reshape = state_rate.view(-1,5,1)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = self.get_mask(len_states, states.shape[1])
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output



def evaluate(model, test_data, device):
    val_dataset = pd.read_csv(os.path.join(data_directory, test_data))
    val_dataset = RecDataset_seq_CTR(val_dataset, max_len)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8)

    with torch.no_grad():
        for j, (seq, history_rating, len_seq, target, target_rating) in enumerate(val_loader):
            model = model.eval()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            history_rating = history_rating.to(device)
            target_rating = target_rating.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, history_rating, len_seq)
            target = target.view((-1, 1))
            # print(model_output.shape, target.shape)
            scores = (nn.Sigmoid()(torch.gather(model_output, 1, target))).view(-1).cpu()
            labels = target_rating.cpu()
            if j == 0:
                scores_all = scores
                labels_all = labels
            else:
                scores_all = torch.cat([scores_all, scores])
                labels_all = torch.cat([labels_all, labels])


    return roc_auc_score(labels_all, scores_all)


class RecDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(temp['seq'])
        len_seq = torch.tensor(temp['len_seq'])
        next = torch.tensor(temp['next'])
        return seq, len_seq, next

    def __len__(self):
        return len(self.data)

class RecDataset_seq_CTR(Dataset):
    def __init__(self, data_df, max_len):
        self.data = data_df
        self.max_len = max_len

    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(np.array(eval(temp['history_item_id']), dtype="int"))
        history_rating = torch.tensor(np.array(eval(temp['history_rating']), dtype="int"))
        # Padding only for taking place, use len_seq for mask while training and evaluating
        pad_len = self.max_len - len(seq)
        len_seq = len(seq)
        if pad_len != 0:
            pad_seq = seq[-1].repeat(pad_len)
            pad_rate = history_rating[-1].repeat(pad_len)
            seq = torch.cat([seq, pad_seq]) 
            history_rating = torch.cat([history_rating, pad_rate]) 

        next = torch.tensor(temp['item_id'])
        next_rating = torch.tensor(temp["rating"])
        
        return seq, history_rating, len_seq, next, next_rating

    def __len__(self):
        return len(self.data)


def main(result_folder, topk=[1,3,5,10,20]):
    print(f"Model: {args.model}")
    if args.model=='SASRec':
        # model = SASRec(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
        model = SASRec_with_label_CTR(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
    if args.model == 'Caser':
        model = Caser_with_label_CTR(args.hidden_factor, item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer.to(device)

    train_data = pd.read_csv(os.path.join(data_directory, 'train.csv'))

    if args.randon_sample_num!=-1:
        train_data = train_data.sample(args.randon_sample_num)

    train_dataset = RecDataset_seq_CTR(train_data, max_len)
    print(f"Train dataset length: {len(train_dataset)}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    print("train_loader created")


    total_step=0
    auc_best = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size) + (int(num_rows/args.batch_size) * args.batch_size != num_rows)
    for i in range(args.epoch):
        # for j in tqdm(range(num_batches)):
        for j, (seq, history_rating, len_seq, target, target_rating) in tqdm(enumerate(train_loader)):
            print(seq, history_rating, len_seq, target, target_rating)
            model = model.train()
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            history_rating = history_rating.to(device)
            target_rating = target_rating.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, history_rating, len_seq)
            target = target.view((-1, 1))
            scores = nn.Sigmoid()(torch.gather(model_output, 1, target))

            # labels = torch.tensor(target_rating.to(device), dtype=float)
            labels = target_rating.to(device).float()

            loss = bce_loss(scores, labels.view(-1,1))

            loss_all = loss
            loss_all.backward()
            optimizer.step()

            if True:

                total_step+=1
                # if total_step % 200 == 0:
                #     print("the loss in %dth step is: %f" % (total_step, loss_all))
                #     # logging.info("the loss in %dth step is: %f" % (total_step, loss_all))

                if total_step % (num_batches * args.eval_num) == 0:

                        # print('VAL PHRASE:')
                        # logging.info('VAL PHRASE:')
                        eval_auc = evaluate(model, 'valid.csv', device)
                        print('TEST PHRASE:')
                        # logging.info('TEST PHRASE:')
                        test_auc = evaluate(model, 'test.csv', device)

                        model = model.train()

                        os.makedirs(result_folder, exist_ok=True)

                        if eval_auc > auc_best:

                            auc_best = eval_auc
                            best_epoch = i
                            early_stop = 0
                            test_result_at_best_eval = test_auc
                            best_model = copy.deepcopy(model)
                            torch.save(best_model, result_folder + "/best_model_" + str(args.l2_decay) + "_" +str(args.seed))


                        else:
                            early_stop += 1
                            if early_stop >= args.early_stop:
                                return best_model, test_result_at_best_eval
                        
                        print('BEST EPOCH:{}'.format(best_epoch))
                        print('EARLY STOP:{}'.format(early_stop))
                        print("test_result_at_best_eval:")
                        print(test_result_at_best_eval)
    return best_model, test_result_at_best_eval
    


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    data_directory = './final_data/' + args.data
    print(f"data_directory: {data_directory}")

    if args.data == "book":
        max_len = 10
        seq_size = max_len  # the length of history to define the seq
        item_num = 271380 # total number of items
    elif args.data == "movie":
        max_len = 20
        seq_size = max_len  # the length of history to define the seq
        item_num = 4000  # total number of items
    elif args.data == "fashion":
        max_len = 4
        seq_size = max_len
        item_num = 6089
    elif args.data == "beauty":
        max_len = 5
        seq_size = max_len
        item_num = 1220

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result_folder = ""
    for path_name in args.result_json_path.split("/")[:-1]:
        result_folder += path_name + "/"

    best_model, test_result_at_best_eval = main(result_folder)

