import csv
import os
import sys
import time

sys.path.append(".")
sys.path.append("..")

import pdb
import json
import _jsonnet # type: ignore
import lightning.pytorch as pl # type: ignore
import pandas as pd
from torch.utils.data import DataLoader # type: ignore
from torchtext.vocab import vocab as build_vocab # type: ignore
from sklearn import metrics

from csv_embedder import PatternTokenizer
from csv_embedder.roberta_finetune_dialect.dataset import RobertaDialectDataset
from csv_embedder.roberta_finetune_dialect.model import RobertaFinetuneDialectDetection
from csv_embedder.roberta_finetune_dialect.data_module import RobertaDialectDataModule
from csv_embedder.roberta_finetune_dialect.tokenizer import RobertaFileTokenizer

from evaluator import Evaluator

import logging
logging.getLogger('lightning').setLevel(0)

class RobertaEvaluator(Evaluator):

    def __init__(self,
                 weights_path,
                 config_path="configs/dialect_roberta.jsonnet",
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

        config = _jsonnet.evaluate_file(self.config_path)
        config = json.loads(config)
        self.batch_size = batch_size
        config["trainer"]["devices"] = [self.cuda_device]
        config["data_module"]["batch_size"] = self.batch_size

        vocabs = {}
        for k in ["delimiter", "quotechar", "escapechar"]:
            tokens = open(config["vocabulary"]["directory"] + f"/dialect_{k}.txt").read().splitlines()
            ordered_tokens =  {t:len(tokens)-i for i,t in enumerate(tokens)}
            token_vocab = build_vocab(ordered_tokens)
            token_vocab.set_default_index(token_vocab["[UNK]"])
            vocabs[k] = token_vocab

        self.vocabs = vocabs
        self.roberta = RobertaFinetuneDialectDetection(vocabs=vocabs,
                                                **config["model"])


        self.roberta.load_weights(weights_path)
        self.roberta.eval()

        del config["data_module"]["n_files"]
        del config["data_module"]["test_data_path"]

        file_tokenizer = RobertaFileTokenizer()
        self.data_module = RobertaDialectDataModule(
            del_vocab = self.vocabs["delimiter"],
            quo_vocab = self.vocabs["quotechar"],
            esc_vocab = self.vocabs["escapechar"],
            tokenizer=file_tokenizer,
            n_files = self.subset,
            test_data_path= self.data_dir,
            **config["data_module"]
        )

        self.data_module.prepare_data()
        self.data_module.setup()
        
        annotation_path =f'{self.data_dir}/dialect_annotations.csv'
        self.annotations_df = pd.read_csv(annotation_path)

        self.trainer = pl.Trainer(**config["trainer"])

    def process_wrapper(self, *args, **kwargs):

        dl = self.data_module.test_dataloader()
        lookup= {k:self.vocabs[k].get_itos() for k in ["delimiter", "quotechar", "escapechar"]}
        batches = self.trainer.predict(self.roberta, dl)


        predicted_delimiter = []
        predicted_quotechar = []
        predicted_escapechar = []
        target_delimiter = []
        target_quotechar = []
        target_escapechar = []
        prediction_time = []

        for batch in batches:
            y, target = batch
            batch_size = y["predicted_delimiter"].shape[0]
            predicted_delimiter += [lookup["delimiter"][i] for i in y["predicted_delimiter"].cpu().numpy()]
            predicted_quotechar += [lookup["quotechar"][i] for i in y["predicted_quotechar"].cpu().numpy()]
            predicted_escapechar += [lookup["escapechar"][i] for i in y["predicted_escapechar"].cpu().numpy()]
            if batch_size > 1:
                predict_time = [y["predict_time"]]*batch_size
            else:
                predict_time = [y["predict_time"]]
            prediction_time.extend(predict_time)

            if type(target["target_delimiter"]) == type(y["predicted_delimiter"]):
                target_delimiter += [lookup["delimiter"][i] for i in target["target_delimiter"].cpu().numpy()]
                target_quotechar += [lookup["quotechar"][i] for i in target["target_quotechar"].cpu().numpy()]
                target_escapechar += [lookup["escapechar"][i] for i in target["target_escapechar"].cpu().numpy()]
            else:
                target_delimiter.extend(target["target_delimiter"])
                target_quotechar.extend(target["target_quotechar"])
                target_escapechar.extend(target["target_escapechar"])

        res = [
            {"filename": self.annotations_df["filename"].values[i],
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
    weights_path = "weights/roberta_dialect.pth"
    for dataset in ["test_augmented", "dev_augmented"]:
        test_evaluator = RobertaEvaluator(data_dir = f"data/dialect_detection/{dataset}/",
                                        sys_name="roberta",
                                        experiment_dir = "results/finetune_dialect/",
                                        dataset = dataset,
                                        augmented=True,
                                        original=True,
                                        subset=None,
                                        n_workers=100,
                                        skip_processing=False,
                                        weights_path=weights_path,
                                        global_max=True,
                                        cuda_device=0,
                                        batch_size=32,
                                        n_repetitions=1,)
        test_evaluator.evaluate()
        test_evaluator.print_results()