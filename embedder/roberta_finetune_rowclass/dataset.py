import copy
import csv
import json
import os
import io
import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from tqdm import tqdm
from typing import List
from csv_embedder.pattern_tokenizer import PatternTokenizer

import multiprocessing

from queue import Empty

def worker(tokenizer, label_vocab, filepath_queue, result_queue, status_queue):
    while True:
        if filepath_queue.empty():
            break
        else:
            try:
                filepath, annotations = filepath_queue.get(timeout=1)
            except Empty:
                break

        empty_idx = [idx for idx, x in enumerate(annotations) if x == "empty"]
        reader = csv.reader(open(filepath, "r", newline=''), delimiter=",", quotechar='"')

        all_rows = []
        all_classes = []
        for idx, row in enumerate(reader):
            if idx in empty_idx:
                continue
            else:
                rowbuf = io.StringIO()
                csv.writer(rowbuf, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL).writerow(row)
                all_rows.append(rowbuf.getvalue()[:-2].replace("\r\n", "\n"))
                all_classes.append(annotations[idx])
        rows = []
        row_classes = []
        if len(all_rows) > tokenizer.max_rows:
            data_idx = [idx for idx, x in enumerate(all_classes) if x == "data"]
            nondata_idx = [idx for idx, x in enumerate(all_classes) if x != "data"]
            if len(nondata_idx) < tokenizer.max_rows-1:
                sampled_idx = data_idx[:(tokenizer.max_rows-len(nondata_idx))]
                for idx,r in enumerate(all_rows):
                    if idx in nondata_idx+list(sampled_idx):
                        rows.append(r)
                        row_classes.append(all_classes[idx])
            else:
                rows = all_rows[:tokenizer.max_rows]
                row_classes = all_classes[:tokenizer.max_rows]
        else:
            rows = all_rows
            row_classes = all_classes

        try:
            assert len(row_classes) == len(rows), f"Number of rows and number of row classes are not the same: {len(row_classes)} != {len(rows)}"
            row_tokens = []
            for r in rows[:tokenizer.max_rows]:
                row_tokens.append(tokenizer(r)["input_ids"].squeeze()[:tokenizer.max_len])
            input_tokens = torch.stack(row_tokens).squeeze()

            row_labels = torch.tensor(label_vocab(row_classes))

            assert len(row_tokens) == len(row_classes)
            assert set(row_classes) != set(["empty"]), f"Row classes are all empty: {row_classes}"

        except Exception as e:
                print(f"Reader exception: {e}")
                print(f"Filename: {filepath}")
                raise e

        result_queue.put([input_tokens.numpy(),row_labels.numpy()])
        status_queue.put(1)


class RobertaRowClassDataset(Dataset):
    def __init__(self, data_path,
                 tokenizer,
                 label_vocab: Vocab,
                 max_rows=10,
                 max_len=32,
                 n_files=None,
                 save_path = None,
                 ):

        self.data_path = data_path
        jsonlines = open(self.data_path).read().splitlines()
        self.annotations = list(x for x in map(json.loads, jsonlines))
        self.annotations = self.annotations[:n_files]
        self.label_vocab = label_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.filepaths = []
        for ann in self.annotations:
            fpath = f"{os.path.dirname(self.data_path)}/{ann['group']}/{ann['filename']}"
            self.filepaths.append(fpath)

        self.input_tokens_list = None
        if save_path is not None:
            self.save_path = save_path
            if os.path.exists(save_path):
                self.load_from_disk(save_path)
            else:
                self.tokenize_and_save(save_path)


    def __len__(self):
        return len(self.annotations)

    def get_groups_indices(self):
        groups = {}
        for idx, ann in enumerate(self.annotations):
            group = ann["group"]
            if group not in groups:
                groups[group] = []
            groups[group].extend(self.annotation_indices[idx])

        return groups

    def load_from_disk(self, path):
        self.input_tokens_list = torch.load(path)
        self.row_tokens = []
        self.row_labels = []
        self.annotation_indices = []
        cur_idx = 0
        for idx, (x0, x1) in enumerate(self.input_tokens_list):
            self.row_tokens.append(x0)
            self.row_labels.append(x1)
            self.annotation_indices.append(list(range(cur_idx, cur_idx+len(x1))))
            cur_idx += len(x1)

        self.row_tokens = torch.vstack(self.row_tokens).view(-1, self.max_len)
        self.row_labels = torch.hstack(self.row_labels).view(-1)

    def __getitem__(self, idx):
        
        if self.input_tokens_list is not None:
            return self.row_tokens[idx,:], self.row_labels[idx]
        else:
            raise NotImplementedError


    def tokenize_and_save(self, path, n_workers = 1):
        n_workers = 4

        if n_workers > 0:
            filepath_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            status_queue = multiprocessing.Queue()


            for idx,filepath in enumerate(self.filepaths):
                ann = self.annotations[idx]["line_annotations"]
                filepath_queue.put((filepath, ann))

            processes = []

            for _ in range(n_workers):
                process = multiprocessing.Process(target=worker, 
                                                    args=(copy.deepcopy(self.tokenizer),
                                                     copy.deepcopy(self.label_vocab),
                                                        filepath_queue, 
                                                        result_queue,
                                                        status_queue))
                processes.append(process)
                process.start()


            status_list = []
            results = []
            with tqdm(total=len(self.filepaths), desc="Tokenizing input") as pbar:
                while any(process.is_alive() for process in processes) or len(status_list) < len(self.filepaths):
                    if status_queue.empty():
                        pass
                    else:
                        x = status_queue.get(timeout=1)
                        status_list.append(x)
                        input_tokens, row_labels = result_queue.get()
                        results.append([torch.tensor(input_tokens), torch.tensor(row_labels)])
                        pbar.update(1)

            while not result_queue.empty():
                input_tokens, row_labels = result_queue.get()
                results.append([torch.tensor(input_tokens), torch.tensor(row_labels)])

            for process in processes:
                process.join()


            filepath_queue.close()
            result_queue.close()

        self.input_tokens_list = results
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.input_tokens_list, path)

        self.row_tokens = []
        self.row_labels = []
        self.annotation_groups = []
        for idx, (x0, x1) in enumerate(self.input_tokens_list):
            self.row_tokens.append(x0)
            self.row_labels.append(x1)
            self.annotation_groups.extend([self.annotations[idx]["group"] for _ in range(len(x1))])


        self.annotation_indices = []
        cur_idx = 0
        for idx, (x0, x1) in enumerate(self.input_tokens_list):
            self.row_tokens.append(x0)
            self.row_labels.append(x1)
            self.annotation_indices.append(list(range(cur_idx, cur_idx+len(x1))))
            cur_idx += len(x1)

        self.row_tokens = torch.vstack(self.row_tokens).view(-1, self.max_len)
        self.row_labels = torch.hstack(self.row_labels).view(-1)