import os
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

MAX_LEN = 512
SEP_TOKEN_ID = 102
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import tqdm
import time
import json
import numpy as np
import random
import torch
from krel import KREL
from sklearn.metrics import f1_score
from table_dataset import TableDataset
from tqdm import tqdm
from tqdm import trange


NERs = {'PERSON1':0, 'PERSON2':1, 'NORP':2, 'FAC':3, 'ORG':4, 'GPE':5, 'LOC':6, 'PRODUCT':7, 'EVENT':8, 'WORK_OF_ART':9, 'LAW':10, 'LANGUAGE':11, 'DATE1':12, 'DATE2':13, 'DATE3':14, 'DATE4':15, 'DATE5':16, 'TIME':17, 'PERCENT':18, 'MONEY':19, 'QUANTITY':20, 'ORDINAL':21, 'CARDINAL':22, 'EMPTY':23}


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', default="clean" ,help='Specify the type argument')
parser.add_argument('--n_folds', default=5, help='Specify the type argument')
args = parser.parse_args()
type_arg = args.type
n_folds = int(args.n_folds)
data_dir = f"../data_{type_arg}"

def setup_seed(seed): # Set up random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_loader(path, batch_size, is_train): # Generate the dataloaders for the training process
    dataset = torch.load(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=dataset.collate_fn)
    loader.num = len(dataset)
    return loader


def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def train_model(model,
                train_loader,
                val_loader,
                lr,
                model_save_path='.pkl',
                early_stop_epochs=5,
                epochs=20): # Training process
    no_improve_epochs = 0
    weight_decay = 1e-2
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    cur_best_v_loss =10.0
    for epoch in range(1,epochs+1):
        
        model.train()
        epoch_loss = 0
        v_epoch_loss = 0
        train_length = 0
        tic = time.time()
        bar1 = tqdm(train_loader)
            
        for i,(ids, rels, subs, labels) in enumerate(bar1):
            labels = labels.cuda()
            rels = rels.cuda()
            subs = subs.cuda()
            output = model(ids.cuda(), rels, subs)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            loss = loss_fn(y_pred_prob.view(-1, model.n_classes), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            length_label = len(labels)
            del ids, rels, subs, labels
            torch.cuda.empty_cache() # Release the memory
        train_length += len(bar1)
        print("Epoch:", epoch, "training_loss:", epoch_loss / (train_length))
        model.eval()
        if val_loader is not None:
            bar2 = tqdm(val_loader)
            pred_labels = []
            true_labels = []
            toc = time.time()
            print('training time:', toc-tic)
            for j, (ids, rels, subs, labels) in enumerate(bar2):
                labels = labels.cuda()
                rels = rels.cuda()
                subs = subs.cuda()
                output = model(ids.cuda(), rels, subs)
                y_pred_prob = output
                y_pred_label = y_pred_prob.argmax(dim=1)
                vloss = loss_fn(y_pred_prob.view(-1, model.n_classes), labels.view(-1))
                pred_labels.append(y_pred_label.detach().cpu().numpy())
                true_labels.append(labels.detach().cpu().numpy())
                v_epoch_loss += vloss.item()
                del ids, rels, subs
                torch.cuda.empty_cache()
            tac = time.time()
            pred_labels = np.concatenate(pred_labels)
            true_labels = np.concatenate(true_labels)
            val_length = len(bar2)
            print("validation_loss:", v_epoch_loss / (val_length))
            f1_scores = metric_fn(pred_labels, true_labels)
            print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
            print('validation time:', tac-toc)
            if v_epoch_loss / (val_length) < cur_best_v_loss:
                torch.save(model.state_dict(),model_save_path)
                cur_best_v_loss = v_epoch_loss / (val_length)
                no_improve_epochs = 0
                print("model updated")
            else:
                no_improve_epochs += 1
            if no_improve_epochs == 5:
                print("early stop!")
                with open(f'../results/renemb_{type_arg}-RECA_lr={lr}_bs={BS}_max={MAX_LEN}_dev_f1.txt', 'a') as f:
                    f.write("\n"+str(f1_scores['weighted_f1']) + '\t' + str(f1_scores['macro_f1']))
                break
        elif epoch == epochs:
            torch.save(model.state_dict(),model_save_path)
            with open(f'../results/renemb_{type_arg}-RECA_lr={lr}_bs={BS}_max={MAX_LEN}_dev_f1.txt', 'a') as f:
                f.write(str("\n"+f1_scores['weighted_f1']) + '\t' + str(f1_scores['macro_f1']))


def test_model(model,test_loader,lr,model_save_path='.pkl'):  
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
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1']


if __name__ == '__main__':
    setup_seed(20)
    with open('./dataset_labels.txt', 'r') as label_file:
        labels = label_file.read().split('\n')
    label_dict = {k: v for v, k in enumerate(labels)}
    
    BS = 8
    lrs = [1e-5]
    print('start loading data')
    
    for k_idx in range(n_folds):
        train_loader_path = f'{data_dir}/tokenized_data/train_'+str(MAX_LEN)+'_fold_'+str(k_idx)
        valid_loader_path = f'{data_dir}/tokenized_data/valid_'+str(MAX_LEN)+'_fold_'+str(k_idx)
        val_loader = get_loader(path=valid_loader_path, batch_size=BS, is_train=False)    
        train_loader = get_loader(path=train_loader_path, batch_size=BS, is_train=True)
        for lr in lrs:
            print('start training fold', k_idx, 'learning rate', lr, 'batch size', BS, 'max length', MAX_LEN)
            model = KREL().cuda()
            model_save_path = f'../checkpoints/renemb_{type_arg}-RECA_lr={lr}_bs={BS}_max={MAX_LEN}_{k_idx}.pkl'
            train_model(model, train_loader, val_loader,lr, model_save_path=model_save_path)
    
    lr = 1e-5
    # Train the model on the whole training set
    train_loader_path = f'{data_dir}/tokenized_data/train_val_'+str(MAX_LEN)
    train_loader = get_loader(path=train_loader_path, batch_size=BS, is_train=True)
    print('start training fold full', 'learning rate', lr, 'batch size', BS, 'max length', MAX_LEN)
    model = KREL().cuda()
    model_save_path = f'../checkpoints/renemb_{type_arg}-RECA_lr={lr}_bs={BS}_max={MAX_LEN}_full.pkl'
    train_model(model, train_loader, None,lr, model_save_path=model_save_path)

    test_loader_path = f'{data_dir}/tokenized_data/test_'+str(MAX_LEN)
    test_loader = get_loader(path=test_loader_path, batch_size=1, is_train=False)
    weighted_f1, macro_f1 = test_model(model, test_loader,lr, model_save_path=model_save_path)

    print("The test F1 score is:", weighted_f1)
    print("The test macro F1 score is:", macro_f1)
