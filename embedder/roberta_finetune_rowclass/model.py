import logging
import os
import pdb
import time
from typing import Dict, Any
import lightning.pytorch as pl

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torchmetrics
from torchtext.vocab import Vocab
from torch import optim

logger = logging.getLogger(__name__)
from transformers import AutoTokenizer, AutoModel, AutoConfig

torch.set_float32_matmul_precision('medium')


class RobertaFinetuneRowClassification(pl.LightningModule):
    def __init__(
        self,
        label_vocab: Vocab,
        n_classes: int,
        save_path: str = "",
        classes_weights: list = [1, 1, 1, 1, 1, 1],
        ignore_class: str = "",
        optimizer_lr=1e-4,
        *args,
        **kwargs
    ):

        super(RobertaFinetuneRowClassification, self).__init__(*args, **kwargs)
        self.label_vocab = label_vocab

        self.model = AutoModel.from_pretrained("xlm-roberta-large")
        # Input: batch_size x channels_img x n_rows x max_tokens
        # Input: B_s x Dx64x64, output: 4x32x32
        self.rowclass_linear = nn.Linear(1024, n_classes)

        (
            self.data_index,
            self.header_index,
            self.metadata_index,
            self.group_index,
            self.derived_index,
            self.notes_index,
            self.empty_index
        ) = self.label_vocab(
            ["data", "header", "metadata", "group", "derived", "notes", "empty"]
        )

        if ignore_class == "data":
            ignore_index = self.data_index
        elif ignore_class == "empty":
            ignore_index = self.empty_index
        else:
            ignore_index = -1
        weight = torch.tensor(classes_weights, dtype=torch.float)
        self.finetune_loss = CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=n_classes, average="macro", ignore_index=ignore_index)

        self.train_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=n_classes, average=None, ignore_index=ignore_index)

        self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=n_classes, average="macro", ignore_index=ignore_index)

        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=n_classes, average=None, ignore_index=ignore_index)

        self.n_classes = n_classes
        self.save_path = save_path
        self.optimizer_lr = optimizer_lr

    def forward(self, input_tokens, row_labels, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the self attention, the file embeddings
        """

        y = self.model(input_tokens)["last_hidden_state"]
        
        # use row_cls
        row_cls = y[:, 0, :]
        rowclass_logits = self.rowclass_linear(row_cls)

        return {
            "row_embeddings": row_cls,
            "lineclass_logits": rowclass_logits,
        }

    def training_step(self, batch, batch_idx):
        x, row_classes = batch
        output = self.forward(x, row_classes)
        lineclass_logits = output["lineclass_logits"]

        target = row_classes.view(-1,)
        predicted = lineclass_logits.view(-1, self.n_classes)

        loss = self.finetune_loss(predicted, target)

        acc = self.train_accuracy(predicted, target)
        f1 = self.train_f1(predicted, target)
        
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_header", f1[self.header_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_metadata", f1[self.metadata_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_data", f1[self.data_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_f1_empty", f1[self.empty_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_group", f1[self.group_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_derived", f1[self.derived_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_notes", f1[self.notes_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, row_classes = batch
        output = self.forward(x, row_classes)
        lineclass_logits = output["lineclass_logits"]

        target = row_classes.view(-1,)
        predicted = lineclass_logits.view(-1, self.n_classes)

        loss = self.finetune_loss(predicted, target)
        acc = self.val_accuracy(predicted, target)
        f1 = self.val_f1(predicted, target)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_header", f1[self.header_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_metadata", f1[self.metadata_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_data", f1[self.data_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_f1_empty", f1[self.empty_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_group", f1[self.group_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_derived", f1[self.derived_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_notes", f1[self.notes_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch: Any, batch_idx: int=0, dataloader_idx: int = 0) -> Any:
        x, target = batch
        start = time.process_time()
        output = self.forward(x)
        y = output["lineclass_logits"]
        z = nn.Softmax(dim=2)(y.type(torch.double))
        output["lineclass"] = torch.argmax(z, dim=2)
        output["predict_time"] = time.process_time()-start
        return output, target
        

    def on_train_epoch_end(self) -> None:
        self.train_accuracy.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
        # optimizer = optim.SGD(self.parameters(), lr=self.optimizer_lr)
        return optimizer

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.macro_accuracy.compute().tolist()
        f1 = self.f1_score.compute().tolist()
        if reset:
            self.macro_accuracy.reset()
            self.f1_score.reset()
        fscores = {
            "f1_data": f1[self.data_index],
            "f1_header": f1[self.header_index],
            "f1_metadata": f1[self.metadata_index],
            "f1_group": f1[self.group_index],
            "f1_derived": f1[self.derived_index],
            "f1_notes": f1[self.notes_index],
            "f1_overall": np.mean(f1),
        }

        return {"macro_accuracy": accuracy, **fscores}

    def save_weights(self):
        print("Saving model in " + self.save_path)
        torch.save(self.state_dict(), self.save_path)


    def load_weights(self, load_path=None):
        if load_path is not None and os.path.isfile(load_path):
            try:
                self.load_state_dict(torch.load(load_path))
            except RuntimeError:
                self.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
            print("Restored model from " + load_path)
        else:
            print("base model not found or load path not given.")