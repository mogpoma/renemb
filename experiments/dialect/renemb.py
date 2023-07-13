import csv
import os
import sys
import time

sys.path.append(".")
sys.path.append("..")

import json
import _jsonnet # type: ignore
import lightning.pytorch as pl # type: ignore
import pandas as pd
from torch.utils.data import DataLoader # type: ignore
from torchtext.vocab import vocab as build_vocab # type: ignore
from sklearn import metrics

from embedder import PatternTokenizer
from embedder.renemb_finetune_dialect.dataset import DialectDataset
from embedder.renemb_finetune_dialect.model import RenembFinetuneDialectDetection

from evaluator import Evaluator

import logging
logging.getLogger('lightning').setLevel(0)

class RenembEvaluator(Evaluator):

    def __init__(self,
                 config_path="configs/dialect.jsonnet",
                 weights_path="",
                 global_max=True,
                 cuda_device=-1,
                 batch_size = 16,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        config = _jsonnet.evaluate_file(self.config_path,
                                        ext_vars={"max_len": "128",
                                                  "encoding_dim": "128"})
        config = json.loads(config)
        self.batch_size = batch_size
        config["trainer"]["devices"] = [self.cuda_device]

        self.max_rows = config["data_module"]["max_rows"]
        self.max_len = config["data_module"]["max_len"]
        config["data_module"]["batch_size"] = self.batch_size

        tokens = open(config["vocabulary"]["directory"] + "/tokens.txt").read().splitlines()
        ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
        self.token_vocab = build_vocab(ordered_tokens)
        self.token_vocab.set_default_index(self.token_vocab["[UNK]"])

        dialect_classes = open(config["vocabulary"]["directory"] + "/dialect_labels.txt").read().splitlines()
        ordered_classes = {c: len(dialect_classes) - i for i, c in enumerate(dialect_classes)}
        self.label_vocab = build_vocab(ordered_classes)

        self.renemb = RenembFinetuneDialectDetection(token_vocab=self.token_vocab,
                                                         label_vocab=self.label_vocab,
                                                         **config["model"])
        self.renemb.load_weights(weights_path)
        self.renemb.eval()

        self.trainer = pl.Trainer(**config["trainer"])

    def process_wrapper(self, *args, **kwargs):

        df = pd.read_csv(self.dialect_file).fillna("")
        df = df[df['filename'].isin(self.files)]

        dataset = DialectDataset(df,
                                 data_path=self.data_dir,
                                 token_vocab=self.token_vocab,
                                 label_vocab=self.label_vocab,
                                 max_rows=self.max_rows,
                                 max_len=self.max_len,
                                 tokenizer=PatternTokenizer(),
                                 for_prediction=True,)


        dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        batches = self.trainer.predict(self.renemb, dl)
        lookup_vocab = {i: k for i, k in enumerate(self.token_vocab.get_itos())}
        unk_idx = self.token_vocab(["[UNK]"])[0]

        lookup_vocab[unk_idx] = ""

        predicted_delimiter = []
        predicted_quotechar = []
        predicted_escapechar = []
        target_delimiter = []
        target_quotechar = []
        target_escapechar = []
        prediction_time = []

        for batch in batches:
            y, target = batch
            predicted_delimiter += [lookup_vocab[i] for i in y["predicted_delimiter"].cpu().numpy()]
            predicted_quotechar += [lookup_vocab[i] for i in y["predicted_quotechar"].cpu().numpy()]
            predicted_escapechar += [lookup_vocab[i] for i in y["predicted_escapechar"].cpu().numpy()]
            prediction_time.append(y["predict_time"])

            if type(target["target_delimiter"]) == type(y["predicted_delimiter"]):
                target_delimiter += [lookup_vocab[i] for i in target["target_delimiter"].cpu().numpy()]
                target_quotechar += [lookup_vocab[i] for i in target["target_quotechar"].cpu().numpy()]
                target_escapechar += [lookup_vocab[i] for i in target["target_escapechar"].cpu().numpy()]
            else:
                target_delimiter.extend(target["target_delimiter"])
                target_quotechar.extend(target["target_quotechar"])
                target_escapechar.extend(target["target_escapechar"])

        res = [
            {"filename": df["filename"].iloc[0],
            "predicted_delimiter": predicted_delimiter[i],
            "predicted_quotechar": predicted_quotechar[i],
            "predicted_escapechar": predicted_escapechar[i],
            "target_delimiter": target_delimiter[i],
            "target_quotechar": target_quotechar[i],
            "target_escapechar": target_escapechar[i],
            "prediction_time": prediction_time[i],
        } for i in range(len(predicted_delimiter))]
        return res

if __name__ == "__main__":
    weights_path = "weights/renemb_dialect.pth"

    print("Test set:")
    test_evaluator = RenembEvaluator(data_dir = "data/dialect_detection/test/",
                                    sys_name="renemb",
                                    experiment_dir = "results/finetune_dialect/",
                                    dataset = "test",
                                    augmented=False,
                                    original=True,
                                    subset=None,
                                    n_workers=100,
                                    skip_processing=False,
                                    weights_path=weights_path,
                                    global_max=True,
                                    cuda_device=0,
                                    batch_size=1,)
    test_evaluator.evaluate()
    test_evaluator.print_results()
 
    print("Difficult set:")
    test_evaluator = RenembEvaluator(data_dir = "data/dialect_detection/difficult/",
                                    sys_name="renemb",
                                    experiment_dir = "results/finetune_dialect/",
                                    dataset = "difficult",
                                    augmented=False,
                                    original=True,
                                    subset=None,
                                    n_workers=100,
                                    skip_processing=False,
                                    weights_path=weights_path,
                                    global_max=True,
                                    cuda_device=0,
                                    batch_size=1,)
    test_evaluator.evaluate()
    test_evaluator.print_results()