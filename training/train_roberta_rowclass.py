import os
import shutil
from pathlib import Path
import _jsonnet
import json
import sys
import pdb
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from torchtext.vocab import vocab

sys.path.append(".")
sys.path.append("..")
from csv_embedder.roberta_finetune_rowclass.model import RobertaFinetuneRowClassification
from csv_embedder.roberta_finetune_rowclass.data_module import RobertaRowClassDataModule
from csv_embedder.roberta_finetune_rowclass.dataset import RobertaRowClassDataset
from csv_embedder.roberta_finetune_rowclass.tokenizer import RobertaRowFileTokenizer

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import pandas as pd

MODEL_FOLDER = "results/rowclass/"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_datasets = [f"cv_{i}" for i in range(10)]
# for validation_dataset in ["saus", "cius","govuk", "deex"]:
for validation_dataset in train_datasets + ["troy","mendeley"]:
    CONFIG_PATH = f"configs/rowclass_roberta.jsonnet"
    config = _jsonnet.evaluate_file(CONFIG_PATH,
                                    ext_vars={"validation_dataset": validation_dataset})
    config = json.loads(config)

    line_classes = open(config["vocabulary"]["directory"] + "/lineclass_labels.txt").read().splitlines()
    label_vocab = vocab({lineclass: 1 for lineclass in line_classes})

    dm = RobertaRowClassDataModule(
        label_vocab=label_vocab,
        tokenizer=RobertaRowFileTokenizer(),
        **config["data_module"]
    )

    dm.prepare_data()
    dm.setup()

    model = RobertaFinetuneRowClassification(label_vocab=label_vocab,
                                            **config["model"])

    callbacks=[]
    if validation_dataset in ("mendeley", "troy"):
        limit_val_batches = 0
    else:
        limit_val_batches = None
        callbacks += [EarlyStopping(**config["callbacks"]["early_stopping"]),
                    RichProgressBar(),
                    ModelCheckpoint(monitor="val_loss", save_top_k=1),]
        
    logger = TensorBoardLogger(**config["logger"], name="roberta_val_"+validation_dataset)
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        limit_val_batches=limit_val_batches,
        callbacks = callbacks,
    )
    trainer.fit(model, dm)
    if validation_dataset != "full":
        trainer.validate(model, dm)
    
    model.save_path = config["model"]["save_path"]
    model.save_weights()
