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

sys.path.append(os.path.abspath("."))
from csv_embedder.roberta_finetune_dialect.model import RobertaFinetuneDialectDetection
from csv_embedder.roberta_finetune_dialect.data_module import RobertaDialectDataModule
from csv_embedder.roberta_finetune_dialect.dataset import RobertaDialectDataset
from csv_embedder.roberta_finetune_dialect.tokenizer import RobertaFileTokenizer

from csv_embedder.callbacks import TBLogger

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import pandas as pd

MODEL_FOLDER = "results/dialect/"


CONFIG_PATH = f"configs/dialect_roberta.jsonnet"

config = _jsonnet.evaluate_file(CONFIG_PATH)
config = json.loads(config)

del_vocab_path = config["vocabulary"]["directory"] +"/dialect_delimiter.txt"
if not  os.path.exists(del_vocab_path):
    train_df = pd.read_csv(config["data_module"]["train_data_path"] + "/dialect_annotations.csv")
    val_df = pd.read_csv(config["data_module"]["val_data_path"] + "/dialect_annotations.csv")
    annotations_df = pd.concat([train_df, val_df]).fillna("[UNK]")
    for k in ["delimiter", "quotechar", "escapechar"]:
        tokens = annotations_df[k].unique().tolist()
        tokens+= ["[UNK]"]
        tokens = list(set(tokens))
        if not os.path.exists(config["vocabulary"]["directory"]):
            os.makedirs(config["vocabulary"]["directory"])
        with open(config["vocabulary"]["directory"] + f"/dialect_{k}.txt", "w") as f:
            f.write("\n".join(tokens))


vocabs = {}
for k in ["delimiter", "quotechar", "escapechar"]:
    tokens = open(config["vocabulary"]["directory"] + f"/dialect_{k}.txt").read().splitlines()
    ordered_tokens =  {t:len(tokens)-i for i,t in enumerate(tokens)}
    token_vocab = build_vocab(ordered_tokens)
    token_vocab.set_default_index(token_vocab["[UNK]"])
    vocabs[k] = token_vocab

file_tokenizer = RobertaFileTokenizer()

dm = RobertaDialectDataModule(
    del_vocab = vocabs["delimiter"],
    quo_vocab = vocabs["quotechar"],
    esc_vocab = vocabs["escapechar"],
    tokenizer=file_tokenizer,
    **config["data_module"]
)

dm.prepare_data()
dm.setup()

model = RobertaFinetuneDialectDetection(vocabs=vocabs,
                                        **config["model"])

logger = TBLogger(**config["logger"])
trainer = pl.Trainer(
    **config["trainer"],
    logger=logger,
    callbacks=[
        EarlyStopping(**config["callbacks"]["early_stopping"]),
        RichProgressBar(),
        ModelCheckpoint(monitor='val_loss', save_top_k=1,)
    ],
)
trainer.fit(model, dm)
