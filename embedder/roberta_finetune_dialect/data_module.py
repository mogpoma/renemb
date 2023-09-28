import csv
import json
import pdb
import os
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import Vocab

from .tokenizer import RobertaFileTokenizer
from .dataset import RobertaDialectDataset


class RobertaDialectDataModule(pl.LightningDataModule):

    def __init__(self, train_data_path: str,
                 val_data_path: str,
                 del_vocab: Vocab,
                 quo_vocab: Vocab,
                 esc_vocab: Vocab,
                 n_files: int,
                 tokenizer: RobertaFileTokenizer,
                 max_rows: int,
                 max_len: int,
                 batch_size: int,
                 num_workers: int,
                 test_data_path: str = "",
                 shuffle: bool = True, 
                 save_path: str = None,):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.del_vocab = del_vocab
        self.quo_vocab = quo_vocab
        self.esc_vocab = esc_vocab

        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.n_files = n_files
        self.tokenizer = tokenizer
        self.data_train = None
        self.data_val = None
        self.save_path = save_path

    def prepare_data(self):
        # prepare is always executed on 1 process on CPU
        # It is not recommended to assign state here (e.g. self.x = y)
        # since it is called on a single process and if you assign states here then they won’t be available for other processes.

        # tokenize
        # save it to disk
        for dataset_path in [self.train_data_path, self.val_data_path]:
            dialect_file = f"{dataset_path}/dialect_annotations.csv"
            if not os.path.isfile(dialect_file):
                print(f"Annotation file not found in {dialect_file}, creating it...")
                dialect_path = f"{dataset_path}/dialect/"

                dialect_list = []
                if not os.path.isdir(dialect_path):
                    os.mkdir(dialect_path)
                for file in os.listdir(dialect_path):
                    if file.endswith(".json"):
                        with open(f"{dialect_path}/{file}") as f:
                            annotation = json.load(f)
                        dialect_list.append({"filename": annotation["filename"], **annotation["dialect"]})
                        if annotation["filename"] not in os.listdir(dataset_path + "/csv/"):
                            print("File", annotation["filename"]," not found in csv dir")
                            raise AssertionError

                df = pd.DataFrame(dialect_list)
                Path(os.path.dirname(dialect_file)).mkdir(parents=True, exist_ok=True)
                df.to_csv(dialect_file, index=False, quoting=csv.QUOTE_ALL)

    def setup(self, stage=None):
        pass
        # raise ValueError("Stage not recognized: ", stage)

    def _common_dataloader(self, data_path, dataset_type="train", shuffle=True):
        dialect_file = f"{data_path}/dialect_annotations.csv"
        annotations_df = pd.read_csv(dialect_file)
        annotations_df = annotations_df.fillna("")

        dataset = RobertaDialectDataset(annotations_df,
                                            data_path=data_path,
                                            del_vocab = self.del_vocab,
                                            quo_vocab = self.quo_vocab,
                                            esc_vocab = self.esc_vocab,
                                            max_len = self.max_len,
                                            tokenizer=self.tokenizer,
                                            n_files=self.n_files, 
                                            save_path=f"{self.save_path}/{dataset_type}_inputs.pt",)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self):
        data_path = self.train_data_path
        return self._common_dataloader(data_path, shuffle=self.shuffle, dataset_type="train")

    def val_dataloader(self):
        data_path = self.val_data_path
        return self._common_dataloader(data_path, shuffle=False, dataset_type="val")

    def test_dataloader(self):
        data_path = self.test_data_path
        dataloader = self._common_dataloader(data_path, shuffle=False, dataset_type="test")
        return dataloader
