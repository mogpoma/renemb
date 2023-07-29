"""This file contains bare Renemb model without a classification head for line classification.
 The input to the classification head are the combination of file encodings with row CLS vectors.
 """

import logging
import os
import pdb
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torchmetrics
from torchtext.vocab import Vocab
from torch import optim
import lightning.pytorch as pl

from embedder.renemb_base.model import RenembBase

logger = logging.getLogger(__name__)


class RenembFinetuneRowClassification(RenembBase):
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
        super(RenembFinetuneRowClassification, self).__init__(*args, **kwargs)
        self.label_vocab = label_vocab

        # Input: batch_size x channels_img x n_rows x max_tokens
        # Input: B_s x Dx64x64, output: 4x32x32
        self.token_linear = nn.Linear(self.d_model, 1)
        self.row_file_linear = nn.Linear(self.encoding_dim+self.max_len, n_classes)

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

    def forward(
        self, input_tokens, row_classes=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """

        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the self attention, the file embeddings
        """

        token_type_ids = torch.zeros_like(input_tokens)
        batch_size = input_tokens.size(0)
        ret = torch.arange(self.max_len, dtype=torch.long).to(input_tokens.device)
        ret = ret.unsqueeze(0).expand_as(
            input_tokens
        )  
        embedding = (
            self.token_embedding(input_tokens)
            + self.position_embedding(ret)
            + self.segment_embedding(token_type_ids)
        )
        row_embeddings = self.norm(embedding)

        batched_row_embeddings = row_embeddings.view(
            batch_size * self.max_rows, self.max_len, -1
        )
        pad_attn_mask = input_tokens.view(batch_size * self.max_rows, -1).data.eq(
            self.padding_index
        )
        row_embeddings = self.encoding_layers(
            batched_row_embeddings, src_key_padding_mask=pad_attn_mask
        )

        zero_mask = pad_attn_mask.unsqueeze(2).expand(-1, -1, self.d_model)
        row_embeddings = row_embeddings.masked_fill(zero_mask, 0)
        row_embeddings = row_embeddings.view(
            batch_size, self.max_rows, self.max_len, -1
        )

        row_embeddings = torch.permute(row_embeddings, (0, 3, 1, 2))
        # return (batch, D, n_rows, n_tokens) ()
        x = self.encoder(row_embeddings).view(batch_size, -1)
        mu = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        file_embedding = mu + std * eps

        row_vecs = row_embeddings.permute(0, 2, 3, 1)
        token_ff = self.token_linear(row_vecs).squeeze(-1)
        cat_file_embedding = file_embedding.unsqueeze(1).repeat((1,self.max_rows,1))
        lineclass_encoding = torch.cat((cat_file_embedding, token_ff), dim=2)
        lineclass_logits = self.row_file_linear(lineclass_encoding)

        # attn : [batch_size, n_heads, d_mode, d_model]
        return {
            "row_embeddings": row_embeddings,
            "file_embedding": file_embedding,
            "lineclass_logits": lineclass_logits,
        }

    def training_step(self, batch, batch_idx):
        x, row_classes = batch
        output = self.forward(x, row_classes)
        lineclass_logits = output["lineclass_logits"]

        target = row_classes.view(-1,)
        predicted = lineclass_logits.view(-1, self.n_classes)

        loss = self.finetune_loss(predicted, target)
        if torch.isnan(loss):
            pdb.set_trace()
        acc = self.train_accuracy(predicted, target)
        f1 = self.train_f1(predicted, target)
        
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_header", f1[self.header_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_metadata", f1[self.metadata_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_data", f1[self.data_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
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

        loss = self.finetune_loss(
            predicted, target
        )  # concatenate rows from different files in one "batch"
        acc = self.val_accuracy(predicted, target)
        f1 = self.val_f1(predicted, target)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_header", f1[self.header_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_metadata", f1[self.metadata_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_data", f1[self.data_index], on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        path = self.save_path + "_current"
        print("Saving model in " + path)
        torch.save(self.state_dict(), path)

        self.train_accuracy.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
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
        if load_path is None:
            load_path = self.save_path
        if os.path.isfile(load_path):
            print("Restoring model from " + load_path)
            try:
                self.load_state_dict(torch.load(load_path))
            except RuntimeError:
                self.load_state_dict(
                    torch.load(load_path, map_location=torch.device("cpu"))
                )
        else:
            print("Renemb base model not found")

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Same as allennlp load_state_dict except that if the weight dictionary comes from a
        super model from Renemb we extract only the base weights WARNING: strong coupling here.
        """
        if len([k for k in state_dict if "renemb" in k]):
            state_dict = {
                k[8:]: v for k, v in state_dict.items() if "renemb" in k
            }  
