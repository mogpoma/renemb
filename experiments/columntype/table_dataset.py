import torch
from torch.utils.data import Dataset

MAX_LEN = 512
SEP_TOKEN_ID = 102

class TableDataset(Dataset):
    # code from train/test
    def __init__(self, 
                 target_cols, 
                 rel_cols, 
                 sub_rel_cols, 
                 tokenizer, 
                 labels=[]):
        self.labels = labels
        self.tokenizer = tokenizer
        self.rel_cols = rel_cols
        self.target_cols = target_cols
        self.sub_rel_cols = sub_rel_cols


        self.labels = []
        self.data = []
        self.rel = []
        self.sub = []

        for i in range(len(labels)):
            self.labels.append(torch.tensor(labels[i]))
            target_token_ids = self.tokenize(target_cols[i])
            self.data.append(target_token_ids)

            if len(rel_cols[i]) == 0: # If there is no related tables, use the target column content
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(rel_cols[i])
            self.rel.append(rel_token_ids)
            
            if len(sub_rel_cols[i]) == 0: # If there is no sub-related tables, use the target column content
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(sub_rel_cols[i])
            self.sub.append(sub_token_ids)
        
    def tokenize(self, text):
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set(self, cols):
        text = ''
        for i, col in enumerate(cols):
            for cell in col:
                text+=cell
                text+=' '
            if not i==len(cols)-1:
                text += '[SEP]'
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids
    
    def tokenize_set_equal(self, cols): # Assigning the tokens equally to each identified column
        init_text = ''
        for i, col in enumerate(cols):
            for cell in col:
                init_text+=cell
                init_text+=' '
            if not i==len(cols)-1:
                init_text += '[SEP]'
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= MAX_LEN:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)     
        else:
            ratio = MAX_LEN/total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col)*ratio:
                        break
                    text += cell
                    text += ' '
                if not i==len(cols)-1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        return self.data[idx], self.rel[idx], self.sub[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels

