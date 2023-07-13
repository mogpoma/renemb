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
from embedder.pattern_tokenizer import PatternTokenizer
from embedder.magritte_finetune_estimate.model import MagritteFinetuneEstimation
from embedder.magritte_finetune_estimate.data_module import EstimationDataModule
from embedder.callbacks import PretrainLoaderCallback, TBLogger

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

MODEL_FOLDER = "results/estimate/"
# remove the folder MODEL_FOLDER/tensorboard
# if os.path.exists(MODEL_FOLDER + "tensorboard/lightning_logs/"):
# shutil.rmtree(MODEL_FOLDER + "tensorboard/lightning_logs/")

CONFIG_PATH = "configs/estimate.jsonnet"

pre_params = [(128, 128)]  # max_len, encoding_dim

# convert the string to a dictionary
for max_len, encoding_dim in pre_params:
    config = _jsonnet.evaluate_file(
        CONFIG_PATH,
        ext_vars={"max_len": str(max_len), "encoding_dim": str(encoding_dim)},
    )
    config = json.loads(config)

    tokens = open(config["vocabulary"]["path"]).read().splitlines()
    tokens[tokens.index("")] = "\n"
    ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
    token_vocab = build_vocab(ordered_tokens)
    token_vocab.set_default_index(token_vocab["[UNK]"])

    model = MagritteFinetuneEstimation(token_vocab=token_vocab, **config["model"])

    dm = EstimationDataModule(
        token_vocab=token_vocab, tokenizer=PatternTokenizer(), **config["data_module"]
    )

    dm.prepare_data()
    dm.setup()

    logger = TBLogger(**config["logger"])
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[
            PretrainLoaderCallback(**config["callbacks"]["pretrain_loader"]),
            EarlyStopping(**config["callbacks"]["early_stopping"]),
            ModelCheckpoint(monitoing="val_loss", save_top_k=1,),
            # RichProgressBar(),
        ],
        # strategy = "deepspeed_stage_2",
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    # trainer.test(model, dm)
