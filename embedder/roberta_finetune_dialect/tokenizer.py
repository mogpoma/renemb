import pdb
import torch
import regex
from typing import List
from transformers import AutoTokenizer


class RobertaFileTokenizer():
    def __init__(
        self,
        max_rows = 128,
        max_len = 510,
        token_vocab=None,
        *args, **kwargs,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.max_rows= max_rows
        self.max_len= max_len
        self.token_vocab = self._tokenizer.get_vocab()
        self.cls_token = self._tokenizer.cls_token
        self.sep_token = self._tokenizer.sep_token
        self.pad_token = self._tokenizer.pad_token
        
    def __call__(self, text: str):
        return self._tokenizer(text, return_tensors="pt", padding="max_length")

    def __len__(self):
        return len(self._tokenizer)

    def tokenize_file(self, csv_path):

            rawdata = open(csv_path, "rb").read()

            try:
                rows = rawdata.decode("utf-8").splitlines()[:self.max_rows]
            except UnicodeDecodeError:
                rows = rawdata.decode("latin-1").splitlines()[:self.max_rows]

            text = self.sep_token.join(rows)
            return self.__call__(text)["input_ids"].squeeze()[:self.max_len]