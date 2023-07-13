import os
import json
import pdb
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
from pqdm.processes import pqdm
import urllib.parse

SUBSET = None
N_JOBS = 100
source_dir = "data/line_classification/"
train_file= "train_dev_annotations.jsonl"
test_file = "test_annotations.jsonl"

def line_annotations_dict(file_dict, ds_name):
    x = file_dict
    filename = f"{x.pop('file_name')}_{x.pop('table_id')}.csv"
    unquoted_name = urllib.parse.unquote(filename)
    unquoted_name = urllib.parse.unquote(unquoted_name)
    unquoted_name = urllib.parse.unquote(unquoted_name)
    x["filename"] = unquoted_name

    del x["table_array"]
    df = pd.DataFrame(x.pop("annotations")).replace("empty", np.nan)
    line_annotations = pd.DataFrame.mode(df, axis=1).replace(np.nan, "empty")
    x["line_annotations"] = line_annotations[0].values.tolist()
    x["group"] = ds_name
    return x

def aggregate_annotations(dataset_names):
    annotations = []
    for jf in dataset_names:
        lines = open(source_dir + jf).read().splitlines()
        ds = list(map(json.loads, lines))
        ds_name = jf.split(".jsonl")[0]

        args = [{"file_dict": d, "ds_name": ds_name} for d in ds]
        ds_annotations = list(pqdm(args, line_annotations_dict, n_jobs=N_JOBS, argument_type="kwargs"))

        assert len([x for x in ds_annotations if type(x) != dict]) == 0, "Processing annotations with errors"
        annotations += ds_annotations
    return annotations


if __name__ == "__main__":
    dataset_files = [x for x in os.listdir(source_dir) if x.endswith(".jsonl")]

    train_files = ["deex.jsonl","saus.jsonl","cius.jsonl", "govuk.jsonl"]
    train_annotations = aggregate_annotations(train_files)
    with open(source_dir+train_file, "w") as f:
        f.writelines(f'{json.dumps(d)}\n' for d in train_annotations)

    test_files = ["mendeley.jsonl","troy.jsonl"]
    test_annotations = aggregate_annotations(test_files)
    with open(source_dir+test_file, "w") as f:
        f.writelines(f'{json.dumps(d)}\n' for d in test_annotations)

