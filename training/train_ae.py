import os
import shutil
import _jsonnet
import json
import sys
sys.path.append(".")
sys.path.append("..")
from embedder.pattern_tokenizer import PatternTokenizer
from embedder.renemb_pretrain_ae.model import RenembPretrainingVAE
from embedder.renemb_pretrain_ae.data_module import CsvFileDataModule
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.loggers import TensorBoardLogger
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from embedder.callbacks import PretrainLoaderCallback

MODEL_FOLDER = "results/pretrain_ae/"
# if os.path.exists(MODEL_FOLDER + "tensorboard"):
    # shutil.rmtree(MODEL_FOLDER + "tensorboard")

CONFIG_PATH = "configs/ae.jsonnet"

for encoding_dim in [32, 64, 128, 256, 512]:
    config = _jsonnet.evaluate_file(CONFIG_PATH, ext_vars = {"encoding_dim":str(encoding_dim)})
    config = json.loads(config)

    tokens = open(config["vocabulary"]["path"]).read().splitlines()
    tokens[tokens.index("")] = "\n"
    ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
    token_vocab = build_vocab(ordered_tokens)
    token_vocab.set_default_index(token_vocab["[UNK]"])

    model = RenembPretrainingVAE(token_vocab=token_vocab, **config["model"])

    dm = CsvFileDataModule(
        token_vocab=token_vocab,
        tokenizer=PatternTokenizer(),
        **config["data_module"])

    dm.prepare_data()
    dm.setup()

    logger = TensorBoardLogger(**config["logger"])
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[
            PretrainLoaderCallback(**config["callbacks"]["pretrain_loader"]),
            # EarlyStopping(**config["callbacks"]["early_stopping"]),
            RichProgressBar(),
            ModelCheckpoint(monitor='val_loss', save_top_k=1,)
        ],
        # limit_val_batches=0,
    )
    trainer.fit(model, dm)
