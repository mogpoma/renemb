import os
import shutil
import pandas as pd
from numpy.random import default_rng

if __name__ == "__main__":

    SEED = 42
    rng = default_rng(seed=SEED)
    TRAIN_RATIO = 0.8

    data_dir  = "data/dialect_detection"
    input_dir = f"{data_dir}/overall_augmented/"
    train_dir = f"{data_dir}/train_augmented/"
    dev_dir   = f"{data_dir}/dev_augmented/"

    df = pd.read_csv(input_dir + "dialect_annotations.csv")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_df = df[:int(len(df) * TRAIN_RATIO)]
    dev_df = df[int(len(df) * TRAIN_RATIO):]

    # create train_dir if it doesn't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(train_dir+"csv/")
        os.makedirs(train_dir+"dialect/")
        os.makedirs(train_dir+"dialect_tags/")

    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)
        os.makedirs(dev_dir+"csv/")
        os.makedirs(dev_dir+"dialect_tags/")

    train_files = train_df["filename"].values
    dev_files = dev_df["filename"].values

    print(f"Number of train files: {len(train_df)}-{len(train_files)}")
    print(train_files[0])

    print("Copying files to train directory...")
    for f in train_files:
        shutil.copy(input_dir+"csv/"+f, train_dir+"csv/"+f)
        shutil.copy(input_dir+"dialect/"+f+"_dialect.json", train_dir+"dialect/"+f+"_dialect.json")
        shutil.copy(input_dir+"dialect_tags/"+f+"_tags.csv", train_dir+"dialect_tags/"+f+"_tags.csv")

    print("Copying files to dev directory...")
    for f in dev_files:
        shutil.copy(input_dir+"csv/"+f, dev_dir+"csv/"+f)
        shutil.copy(input_dir+"dialect_tags/"+f+"_tags.csv", dev_dir+"dialect_tags/"+f+"_tags.csv")

    train_df.to_csv(train_dir + "dialect_annotations.csv", index=False)
    dev_df.to_csv(dev_dir + "dialect_annotations.csv", index=False)
    print("Done.")