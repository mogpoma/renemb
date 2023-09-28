import csv
import json
import os
import io
import pdb
import os
import traceback

import chardet
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab # type: ignore
import sys
sys.path.append(".")
sys.path.append("..")

from csv_embedder.renemb_base.dataset import CsvFileDataset
from .tokenizer import RobertaFileTokenizer
from torch.utils.data import Dataset


class RobertaDialectDataset(Dataset):

    def __init__(self, 
                 annotations_df,
                 del_vocab: Vocab,
                 quo_vocab: Vocab,
                 esc_vocab: Vocab,
                 data_path:str,
                 save_path: str,
                 tokenizer: RobertaFileTokenizer,
                 n_files=None,
                 max_len=512,
                 for_prediction=False,
                 ):

        super(RobertaDialectDataset, self).__init__()
        self.save_path = save_path

        self.input_dataset = CsvFileDataset(
            filepaths=[f"{data_path}/csv/{f['filename']}" for f in annotations_df.to_dict("records")],
            tokenizer=tokenizer,
            token_vocab=tokenizer.token_vocab,
            save_path = save_path,
            subset = n_files
        )

        self.del_vocab = del_vocab
        self.quo_vocab = quo_vocab
        self.esc_vocab = esc_vocab
        self.data_path = data_path
        self.annotations = annotations_df[:n_files].fillna("[UNK]").to_dict("records")
        self.max_len = max_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        filename = ann["filename"]
        delimiter = ann["delimiter"] or "[UNK]"
        quotechar = ann["quotechar"] or "[UNK]"
        escapechar = ann["escapechar"] or "[UNK]"

        input_tokens = self.input_dataset[idx]
        delimiter = self.del_vocab.get_stoi()[delimiter]
        quotechar = self.quo_vocab.get_stoi()[quotechar]
        escapechar = self.esc_vocab.get_stoi()[escapechar]
        
        labels = {"target_delimiter": delimiter,
                  "target_quotechar": quotechar,
                  "target_escapechar": escapechar}
        
        return input_tokens, labels
    
    def collate_fn(self,batch):
        input_tokens_batch, labels_batch = zip(*batch)

        input_tokens_batch = torch.stack([t.squeeze()[:self.max_len] for t in input_tokens_batch])
        
        delimiter_labels = torch.LongTensor([item["target_delimiter"] for item in labels_batch])
        quotechar_labels = torch.LongTensor([item["target_quotechar"] for item in labels_batch])
        escapechar_labels = torch.LongTensor([item["target_escapechar"] for item in labels_batch])

        return input_tokens_batch, {            
            "target_delimiter": delimiter_labels,
            "target_quotechar": quotechar_labels,
            "target_escapechar": escapechar_labels
        }
