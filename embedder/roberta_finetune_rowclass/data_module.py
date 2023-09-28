import pdb

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import Vocab

from .dataset import RobertaRowClassDataset
from .tokenizer import RobertaRowFileTokenizer


class RobertaRowClassDataModule(pl.LightningDataModule):

    def __init__(self, data_path:str,
                 label_vocab: Vocab,
                 n_files:int,
                 tokenizer:RobertaRowFileTokenizer,
                 max_rows: int,
                 max_len: int,
                 batch_size:int,
                 num_workers:int,
                 train_datasets:list,
                 val_dataset_name:str,
                 save_path:str,
                 shuffle:bool = True,):
        super().__init__()
        self.data_path = data_path
        self.label_vocab = label_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_datasets = train_datasets
        self.val_dataset_name = val_dataset_name
        self.shuffle = shuffle
        self.n_files = n_files
        self.tokenizer = tokenizer
        self.data_train = None
        self.data_val = None
        self.save_path = save_path

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            self.dataset_full = RobertaRowClassDataset(self.data_path,
                                            label_vocab = self.label_vocab,
                                            max_rows = self.max_rows,
                                            max_len = self.max_len,
                                            tokenizer = self.tokenizer,
                                            # n_files = self.n_files,
                                            save_path=f"{self.save_path}/rowclass_inputs.pt",)


            group_indices = self.dataset_full.get_groups_indices()
            self.val_indices = group_indices[self.val_dataset_name]
            self.train_indices = []
            for d in self.train_datasets:
                self.train_indices.extend(group_indices[d])
            self.data_train, self.data_val = Subset(self.dataset_full, self.train_indices[:self.n_files]), \
            Subset(self.dataset_full, self.val_indices[:self.n_files])
            

    def _common_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._common_dataloader(self.data_train, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._common_dataloader(self.data_val, shuffle=False)

    def full_dataloader(self):
        return self._common_dataloader(self.dataset_full, shuffle=False)
