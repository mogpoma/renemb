import pdb
import pandas as pd
import os
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

MAX_LEN = 512
SEP_TOKEN_ID = 102
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

CV_FOLDS = 10
N_REPETITIONS = 3
N_WORKERS = 32

import tqdm
import numpy as np
import random
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from tqdm import tqdm
from math import sqrt
from krel import KREL
from table_dataset import TableDataset

NERs = {'PERSON1':0, 'PERSON2':1, 'NORP':2, 'FAC':3, 'ORG':4, 'GPE':5, 'LOC':6, 'PRODUCT':7, 'EVENT':8, 'WORK_OF_ART':9, 'LAW':10, 'LANGUAGE':11, 'DATE1':12, 'DATE2':13, 'DATE3':14, 'DATE4':15, 'DATE5':16, 'TIME':17, 'PERCENT':18, 'MONEY':19, 'QUANTITY':20, 'ORDINAL':21, 'CARDINAL':22, 'EMPTY':23}


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', default="clean" ,help='Specify the type argument')
parser.add_argument('--n_folds', default=5, help='Specify the type argument')
args = parser.parse_args()
type_arg = args.type
n_folds = int(args.n_folds)
data_dir = f"../data_{type_arg}"


def load_jsonl(jsonl_path, label_dict):
    target_cols = []
    labels = []
    rel_cols = []
    sub_rel_cols = []
    one_hot = []
    headers_alias = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    mapping = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}
    with open(jsonl_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            target_cols.append(np.array(item['content'])[:,int(item['target'])])
            target_alias = headers_alias[int(item['target'])]
            labels.append(int(label_dict[item['label']]))
            cur_rel_cols = []
            cur_sub_rel_cols = []
            for rel_col in item['related_cols']:
                cur_rel_cols.append(np.array(rel_col))
            for sub_rel_col in item['sub_related_cols']:
                cur_sub_rel_cols.append(np.array(sub_rel_col))
            rel_cols.append(cur_rel_cols)
            sub_rel_cols.append(cur_sub_rel_cols)
    return target_cols, rel_cols, sub_rel_cols, labels


def get_loader(path, batch_size, is_train):
    dataset = torch.load(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=N_WORKERS, collate_fn=dataset.collate_fn)
    loader.num = len(dataset)
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def test_model(model,test_loader,model_save_path='.pkl'):  
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    bar = tqdm(test_loader)
    pred_labels = []
    true_labels = []
    for i, (ids, rels, subs, labels) in enumerate(bar):
        labels = labels.cuda()
        rels = rels.cuda()
        subs = subs.cuda()
        output = model(ids.cuda(), rels, subs)
        y_pred_prob = output
        y_pred_label = y_pred_prob.argmax(dim=1)
        pred_labels.append(y_pred_label.detach().cpu().numpy())
        true_labels.append(labels.detach().cpu().numpy())
        del ids, rels, subs
        torch.cuda.empty_cache()
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    f1_scores = metric_fn(pred_labels, true_labels)
    # print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1']

if __name__ == '__main__':
    setup_seed(20)
    with open('./dataset_labels.txt', 'r') as label_file:
        labels = label_file.read().split('\n')
    label_dict = {k: v for v, k in enumerate(labels)}

    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_dict = {v : k for k, v in label_dict.items()}
    model = KREL().cuda()
    model_config = "RECA_lr=1e-05_bs=8_max=512"

    batch = 16
    N_REPETITIONS = 2
    dicts = {}

    for cur_rep in range(N_REPETITIONS):
        model_path = f'../checkpoints/renemb_{type_arg}-{model_config}_full.pkl'
        test_loader_path = f'{data_dir}/tokenized_data/test_'+str(MAX_LEN)
        test_loader = get_loader(path=test_loader_path, batch_size=batch, is_train=False)

        cur_w, cur_m = test_model(model, test_loader, model_save_path=model_path)

        wf1 = dicts.get("w_f1_test",[])
        wf1.append(cur_w)
        dicts["w_f1_test"] = wf1

        mf1 = dicts.get("m_f1_test",[])
        mf1.append(cur_m)
        dicts["m_f1_test"] = mf1

    df = pd.DataFrame(dicts)
    df["w_f1_dev_mean"] = df[[col for col in df.columns if "w_f1_dev" in col]].mean(axis=1)
    df["m_f1_dev_mean"] = df[[col for col in df.columns if "m_f1_dev" in col]].mean(axis=1)

    df = df[["w_f1_dev_mean", "m_f1_dev_mean", "w_f1_test", "m_f1_test"]]
    print("Mean:", df.mean(axis=0))
    print("STD:", df.std(axis=0))
