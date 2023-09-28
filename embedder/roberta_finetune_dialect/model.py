
import logging
import os
import pdb
import re
import time
from typing import Dict, Any
import lightning.pytorch as pl

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torchmetrics # type: ignore
from torchmetrics.classification import F1Score # type: ignore

from torchtext.vocab import Vocab # type: ignore
from torch import optim

from embedder.utils import confusion_matrix_figure, f1_table

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

torch.set_float32_matmul_precision('high')


class RobertaFinetuneDialectDetection(pl.LightningModule):

    def __init__(self,
                vocabs: dict[str, Vocab],
                optimizer_lr=1e-4,
                save_path: str = None,
                *args, **kwargs
                ):
        super(RobertaFinetuneDialectDetection, self).__init__(*args, **kwargs)
        self.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained("xlm-roberta-large")
        self.model.to(self.cuda_device)

        self.vocabs = vocabs
        self.linear_del = nn.Linear(1024, len(self.vocabs["delimiter"])).to(self.cuda_device)
        self.linear_quo = nn.Linear(1024, len(self.vocabs["quotechar"])).to(self.cuda_device)
        self.linear_esc = nn.Linear(1024, len(self.vocabs["escapechar"])).to(self.cuda_device)

        self.finetune_loss = nn.CrossEntropyLoss()
        self.delimiter_loss = nn.CrossEntropyLoss()
        self.quotechar_loss = nn.CrossEntropyLoss()
        self.escapechar_loss = nn.CrossEntropyLoss()


        self.train_f1 = {}
        self.val_f1 = {}
        for k in ["delimiter", "quotechar", "escapechar"]:
            self.train_f1[f"f1_{k}"] = F1Score("multiclass", num_classes=len(self.vocabs[k]), average="micro").to(self.cuda_device)
            self.val_f1[f"f1_{k}"] = F1Score("multiclass", num_classes = len(self.vocabs[k]), average="micro").to(self.cuda_device)


        acc_keys = ["train_predicted_delimiter", "train_predicted_quotechar", "train_predicted_escapechar",
                    "train_target_delimiter", "train_target_quotechar", "train_target_escapechar",
                    "val_predicted_delimiter", "val_predicted_quotechar", "val_predicted_escapechar",
                    "val_target_delimiter", "val_target_quotechar", "val_target_escapechar"]

        self.accumulators = {key: torchmetrics.CatMetric() for key in acc_keys}
        self.optimizer_lr = optimizer_lr
        self.save_path = save_path

    def extract_logits(self, input_tokens, tag_softmax, class_idx, padding_mask, vocab_size, batch_size):
        mask = (tag_softmax.argmax(dim=3).eq(class_idx))
        mask = torch.where(~padding_mask, mask, 0).bool()
        detected_tokens = torch.where(mask, input_tokens, self.unk_idx).view(batch_size, -1)

        # this function sets the logits of [UNK] to 0 if there is at least one detected character in the row
        masking = (detected_tokens == self.unk_idx).all(dim=1)
        logits = torch.stack([torch.bincount(row, minlength=vocab_size) for row in detected_tokens], dim=0)
        logits[:, self.unk_idx] *= masking
        logits = torch.softmax(logits.float(), dim=1)
        return logits


    def forward(self, input_tokens, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the self attention, the file embeddings
        """
        input_tokens = input_tokens.type(torch.LongTensor).to(self.cuda_device)
        y = self.model(input_tokens, *args, **kwargs)["last_hidden_state"]
        delimiter_logits = self.linear_del(y[:,0,:]) # cls tokens
        quotechar_logits = self.linear_quo(y[:,0,:]) # cls tokens
        escapechar_logits = self.linear_esc(y[:,0,:]) # cls tokens  

        return {"cls_encoding": y[:,0,:],
                "delimiter_logits": delimiter_logits,
                "quotechar_logits": quotechar_logits,
                "escapechar_logits": escapechar_logits,
                "predicted_delimiter": delimiter_logits.argmax(dim=1),
                "predicted_quotechar": quotechar_logits.argmax(dim=1),
                "predicted_escapechar": escapechar_logits.argmax(dim=1),
                }

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        for x in tqdm_dict.keys():
            if "f1" in x:
                tqdm_dict[x] = round(tqdm_dict[x], 3)
        return tqdm_dict

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self.forward(x)
        delimiter_logits = output["delimiter_logits"]
        quotechar_logits = output["quotechar_logits"]
        escapechar_logits = output["escapechar_logits"]

        del_loss = self.delimiter_loss(delimiter_logits, target["target_delimiter"])
        quote_loss = self.quotechar_loss(quotechar_logits, target["target_quotechar"])
        escape_loss = self.escapechar_loss(escapechar_logits, target["target_escapechar"])
        loss = del_loss + quote_loss + escape_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("del_loss", del_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("quote_loss", quote_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("escape_loss", escape_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for k in ["delimiter", "quotechar", "escapechar"]:
            self.accumulators["train_predicted_" + k].update(output["predicted_" + k])
            self.accumulators["train_target_" + k].update(target["target_" + k])
            f1 = self.train_f1["f1_" + k](output["predicted_" + k], target["target_" + k])
            self.log("f1_"+k, f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        path = self.save_path + "_current.pth"
        print("Saving model in " + path)
        torch.save(self.state_dict(), path)

        [self.train_f1[k].reset() for k in self.train_f1.keys()]

        keys = ["train_predicted_delimiter", "train_predicted_quotechar", "train_predicted_escapechar",
                "train_target_delimiter", "train_target_quotechar", "train_target_escapechar"]
        y_del, y_quo, y_esc, t_del, t_quo, t_esc = [self.accumulators[k].compute().cpu().numpy() for k in keys]

        for k in keys:
            self.accumulators[k].reset()

        tensorboard = self.logger.experiment
        del_cm = confusion_matrix_figure(y_del, t_del, self.vocabs["delimiter"])
        quo_cm = confusion_matrix_figure(y_quo, t_quo, self.vocabs["quotechar"])
        esc_cm = confusion_matrix_figure(y_esc, t_esc, self.vocabs["escapechar"])

        tensorboard.add_figure("train_del_CM", del_cm, self.current_epoch)
        tensorboard.add_figure("train_quo_CM", quo_cm, self.current_epoch)
        tensorboard.add_figure("train_esc_CM", esc_cm, self.current_epoch)

        table = f1_table({"delimiter":y_del,"quotechar":y_quo, "escapechar":y_esc},
                         {"delimiter":t_del,"quotechar":t_quo, "escapechar":t_esc})
        tensorboard.add_text("train_f1_table", table, self.current_epoch)


    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self.forward(x)
        delimiter_logits = output["delimiter_logits"]
        quotechar_logits = output["quotechar_logits"]
        escapechar_logits = output["escapechar_logits"]

        del_loss = self.delimiter_loss(delimiter_logits, target["target_delimiter"])
        quote_loss = self.quotechar_loss(quotechar_logits, target["target_quotechar"])
        escape_loss = self.escapechar_loss(escapechar_logits, target["target_escapechar"])
        loss = del_loss + quote_loss + escape_loss


        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_del_loss", del_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_quote_loss", quote_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_escape_loss", escape_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for k in ["delimiter", "quotechar", "escapechar"]:
            self.accumulators["val_predicted_" + k].update(output["predicted_" + k])
            self.accumulators["val_target_" + k].update(target["target_" + k])
            f1 = self.val_f1["f1_" + k](output["predicted_" + k], target["target_" + k])
            self.log("val_f1_"+k, f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):

        [self.val_f1[k].reset() for k in self.val_f1.keys()]

        keys = ["val_predicted_delimiter", "val_predicted_quotechar", "val_predicted_escapechar",
                "val_target_delimiter", "val_target_quotechar", "val_target_escapechar"]
        y_del, y_quo, y_esc, t_del, t_quo, t_esc = [self.accumulators[k].compute().cpu().numpy() for k in keys]
        for k in keys:
            self.accumulators[k].reset()

        del_cm = confusion_matrix_figure(y_del, t_del, self.vocabs["delimiter"])
        quo_cm = confusion_matrix_figure(y_quo, t_quo, self.vocabs["quotechar"])
        esc_cm = confusion_matrix_figure(y_esc, t_esc, self.vocabs["escapechar"])

        tensorboard = self.logger.experiment
        tensorboard.add_figure("val_del_CM", del_cm, self.current_epoch)
        tensorboard.add_figure("val_quo_CM", quo_cm, self.current_epoch)
        tensorboard.add_figure("val_esc_CM", esc_cm, self.current_epoch)

        table = f1_table({"delimiter":y_del,"quotechar":y_quo, "escapechar":y_esc},
                         {"delimiter":t_del,"quotechar":t_quo, "escapechar":t_esc})
        tensorboard.add_text("val_f1_table", table, self.current_epoch)

    def predict_step(self, batch: Any, batch_idx: int=0, dataloader_idx: int = 0) -> Any:
        start = time.process_time()
        x, target = batch
        output = self.forward(x)
        output["predict_time"] = time.process_time()-start
        return output, target

    def on_train_end(self):
        print("Saving model in " + self.save_path)
        torch.save(self.state_dict(), self.save_path)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
        return optimizer

    def load_weights(self, load_path=None):
        if load_path is not None and os.path.isfile(load_path):
            try:
                self.load_state_dict(torch.load(load_path))
            except RuntimeError:
                self.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
            print("Restored model from " + load_path)
        else:
            print("Base model not found or load path not given.")

    def init_weights(self):
        self.model.init_weights()
