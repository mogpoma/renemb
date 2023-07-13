import os

MAX_LEN = 512
SEP_TOKEN_ID = 102

import tqdm
import time
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from tqdm import trange
from math import sqrt

import sys
sys.path.append('../experiment/')
from table_dataset import TableDataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', default="clean" ,help='Specify the type argument')
parser.add_argument('--n_folds', default=5, help='Specify the type argument')
args = parser.parse_args()
type_arg = args.type
n_folds = int(args.n_folds)
data_dir = f"../data_{type_arg}"


def get_label_dict(path='../experiment/dataset_labels.txt'):
    label_dict = {}
    with open(path, 'r') as label_file:
        labels = label_file.readlines()
    return {label.strip(): i for i, label in enumerate(labels)}
    # with open(path, 'r') as label_file:
        # label_dict = json.load(label_file)
    # return label_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_file(data_path, label_dict):
    labels = []
    out_data = []
    rel_cols = []
    sub_cols = []
    with open(data_path, "r+", encoding="utf8") as jl:
        for item in tqdm(jsonlines.Reader(jl)):
            label_idx = int(label_dict[item['label']])
            target_data = np.array(item['content'])[:,int(item['target'])]
            data = ""
            for i, cell in enumerate(target_data):
                data+=cell
                data+=' '
            cur_rel_cols = []
            cur_sub_rel_cols = []
            for rel_col in item['related_cols']:
                cur_rel_cols.append(np.array(rel_col))
            for sub_rel_col in item['sub_related_cols']:
                cur_sub_rel_cols.append(np.array(sub_rel_col))
            sub_cols.append(cur_sub_rel_cols)
            rel_cols.append(cur_rel_cols)
            labels.append(label_idx)
            out_data.append(data)
    return out_data, rel_cols, sub_cols, labels


if not os.path.exists(f"{data_dir}/tokenized_data/"):
    os.makedirs(f"{data_dir}/tokenized_data/")

if __name__ == '__main__':
    setup_seed(20)
    data_path_train = f'{data_dir}/jsonl_data/train_val_hard_jaccard_ranking.jsonl'
    data_path_test = f'{data_dir}/jsonl_data/test_hard_jaccard_ranking.jsonl'
    label_dict = get_label_dict()
    train_data, train_rel,train_sub, train_labels = load_file(data_path_train, label_dict)
    test_data, test_rel,test_sub, test_labels = load_file(data_path_test, label_dict)
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sfolder_cv = StratifiedKFold(n_splits=n_folds, random_state = 0, shuffle=True)
    for cur_fold, (train_idx, val_idx) in enumerate(sfolder_cv.split(train_data, train_labels)):
        print(f'start loading data, folder {cur_fold}')
        train_col_fold = [train_data[t_idx] for t_idx in train_idx]
        train_rel_fold = [train_rel[t_idx] for t_idx in train_idx]
        train_sub_fold = [train_sub[t_idx] for t_idx in train_idx]
        train_labels_splited = [train_labels[t_idx] for t_idx in train_idx]

        ds_df = TableDataset(train_col_fold, train_rel_fold, train_sub_fold, Tokenizer, train_labels_splited)
        torch.save(ds_df, f'{data_dir}/tokenized_data/train_'+str(MAX_LEN)+'_fold_'+str(cur_fold))

        valid_cols = [train_data[v_idx] for v_idx in val_idx]
        valid_rels = [train_rel[v_idx] for v_idx in val_idx]
        valid_subs = [train_sub[v_idx] for v_idx in val_idx]
        valid_labels_splited = [train_labels[v_idx] for v_idx in val_idx]

        ds_df_v = TableDataset(valid_cols, valid_rels, valid_subs, Tokenizer, valid_labels_splited)
        torch.save(ds_df_v, f'{data_dir}/tokenized_data/valid_'+str(MAX_LEN)+'_fold_'+str(cur_fold))

    ds_df_train_val = TableDataset(train_data, train_rel, train_sub, Tokenizer, train_labels)
    torch.save(ds_df_train_val, f'{data_dir}/tokenized_data/train_val_'+str(MAX_LEN))

    ds_df_t = TableDataset(test_data, test_rel, test_sub, Tokenizer, test_labels)
    torch.save(ds_df_t, f'{data_dir}/tokenized_data/test_'+str(MAX_LEN))
    
