"""
This script can be used to augment dialects data files.
First, it reads the dialect annotations, and creates a list of unique dialects.
Then, for every file it creates a matrix of "augmentability" which contains 1 if the file can be augmented with the dialect, and 0 otherwise.
Then, it creates a new folder for every dialect, and copies the files that can be augmented with that dialect.
"""
import copy
import io
import itertools
import random
import re
import sys
import traceback
import csv

csv.field_size_limit(sys.maxsize)
import _csv
import numpy as np

import os
import pdb
import json
from io import StringIO

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MAX_ROWS = 1024
import pandas as pd
import numpy as np
import argparse
from pqdm.processes import pqdm
from typing import List
from functools import reduce


def list_to_str(l):
    return reduce(lambda x, y=None: str(x) + "\n" + str(y), l)


def get_file_dialect(dialect_path):
    """
    Generates a dataframe with all the dialects and saves it as a csv in dialect_dir
    :param dialect_dir: str
    :return:
    """
    with open(os.path.join(dialect_path), "r") as dialect_file:
        dialect_dict = json.load(dialect_file)["dialect"]
    delimiter = dialect_dict["delimiter"]
    if delimiter == "":
        delimiter = "n"
    quotechar = dialect_dict["quotechar"]
    if quotechar == "":
        quotechar = "n"
    escapechar = dialect_dict["escapechar"]
    if escapechar == "":
        escapechar = "n"
    elif escapechar == '""':
        escapechar = '"'
    dialect = delimiter + quotechar + escapechar
    if delimiter == "n":
        if quotechar == "n":
            if escapechar == "n":
                dialect_type = "000"
            else:
                dialect_type = "001"
        else:
            if escapechar == "n":
                dialect_type = "010"
            else:
                dialect_type = "011"
    elif quotechar == "n" and escapechar == "n":
        dialect_type = "100"
    elif quotechar != "n":
        if escapechar == "n":
            dialect_type = "110"
        else:
            dialect_type = "111"
    else:
        raise Exception(f"Dialect type not recognized, file {dialect_path}")
    if len(dialect) > 3:
        print(dialect_path[:-13] + " has a dialect with more than 3 characters")
        print(f"\t |{delimiter}| |{quotechar}| |{escapechar}|")
        return {"filename": os.path.basename(dialect_path[:-13]), 
                "dialect": dialect, 
                "delimiter": dialect_dict["delimiter"][0], 
                "quotechar": dialect_dict["quotechar"][0],
                "escapechar": dialect_dict["escapechar"][0], 
                "dialect_type": dialect_type}
    return {"filename": os.path.basename(dialect_path[:-13]), "dialect": dialect, "delimiter": dialect_dict["delimiter"], "quotechar": dialect_dict["quotechar"],
                  "escapechar": dialect_dict["escapechar"], "dialect_type": dialect_type}


def parse_content(filepath, dialect, max_rows=128):
    # print(filename, f" del:|{source_del}| quo:|{source_quo}| esc:|{source_esc}|")
    delm = dialect["delimiter"] if (dialect["delimiter"] != "n") else None
    quote = dialect["quotechar"] if (dialect["quotechar"] != "n" and dialect["quotechar"] != "") else None

    if dialect["escapechar"] in ("n", ""):
        escape = None
        dquote = False
    elif dialect["escapechar"] == '"':
        escape = None
        dquote = True
    else:
        escape = dialect["escapechar"]
        dquote = False
    if len(dialect["delimiter"]) > 1:
        print(f"{filepath} delm:|{delm}|")
        return "Error parsing file", pd.DataFrame()
    if len(dialect["quotechar"]) > 1:
        print(f"{filepath} quote:|{quote}|")
        return "Error parsing file", pd.DataFrame()
    if escape:
        if len(escape) > 1:
            print(f"{filepath} escape:|{escape}|")
            return "Error parsing file", pd.DataFrame()

    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            file_content = f.read()
    except UnicodeDecodeError as e:
        with open(filepath, "r", encoding="latin-1") as f:
            file_content = f.read()
        with open(filepath, "w", encoding="utf-8-sig") as f:
            f.write(file_content)
    try:
        if (not delm):
            if quote:
                if not escape:
                    csvreader = [[x.replace(quote, '')] for x in file_content.splitlines()]
                else:
                    co = file_content.replace(f'{escape}{quote}', quote)
                    co = re.sub(rf'^{quote}', '', co)
                    co = re.sub(rf'{quote}\n', '\n', co)
                    csvreader = [[x] for x in co.splitlines()]
            else:
                csvreader = [[x] for x in file_content.splitlines()]
        else:
            csvreader = csv.reader(StringIO(file_content), delimiter=delm, quotechar=quote, escapechar=escape, doublequote=dquote)
    except (TypeError, pd.errors.ParserError, _csv.Error, pd.errors.EmptyDataError) as e:
        print(os.path.basename(filepath), e)
        print(f"\t dialect: |{delm}| |{quote}| |{escape}|")
        return "Error parsing file", pd.DataFrame()

    file_cells = []
    for row in list(csvreader):
        if len(row):
            file_cells.append(row)
        if len(file_cells) > max_rows:
            break

    return {os.path.basename(filepath): {"content": file_content, "file_cells": file_cells}}


def is_augmentable(file_content: str, file_cells: List[List[str]], source_dialect: dict, target_dialect: dict) -> bool:
    """
    Checks if a file is augmentable with the following logic:
    The file is NOT augmentable wrt. to the target dialect if:
        the delimiter is different and the file contains the target delimiter
        the quotechar is different and the file contains the target quotechar
        the escapechar is different and the file contains the target escapechar
    Otherwise, the file IS augmentable if:
        the delimiter is the same, or in case it is not, the file does not contain the target delimiter
        the quotechar is the same, or in case it is not, the file does not contain the target quotechar
        the escapechar is the same, or in case it is not, the file does not contain the target escapechar

    :param filepath: path of the file to check
    :param source_dialect: a dictionary containing {"delimiter": str, "quotechar": str, "escapechar": str, "dialect_type": str}
    :param target_dialect: a dictionary containing {"delimiter": str, "quotechar": str, "escapechar": str, "dialect_type": str}
    :return: bool
    """

    if source_dialect == target_dialect:
        return False

    same_del = (source_dialect["delimiter"] == target_dialect["delimiter"])
    same_quo = (source_dialect["quotechar"] == target_dialect["quotechar"])
    same_esc = (source_dialect["escapechar"] == target_dialect["escapechar"])

    exists_del = (target_dialect["delimiter"] in file_content) if target_dialect["delimiter"] else False
    exists_quo = (target_dialect["quotechar"] in file_content) if target_dialect["quotechar"] else False
    exists_esc = (target_dialect["escapechar"] in file_content) if target_dialect["escapechar"] else False

    if target_dialect["dialect_type"] in ("000", "010", "011"):
        return ((not exists_del) and (not exists_quo) and (not exists_esc))

    if source_dialect["dialect_type"] in ("000", "010", "011"):
        return False  # target has a delimiter, but source does not. We cannot add new data.

    # as long as it's not the same dialect character, and it's not in the file, we can augment
    if target_dialect["dialect_type"] == "111":
        return ((not exists_del) and (not exists_quo) and (not exists_esc))\
                and ((not same_del) or (not same_quo) or (not same_esc) )
    elif target_dialect["dialect_type"] == "110":
        return ((not exists_del) and (not exists_quo))\
                and ((not same_del) or (not same_quo))
    elif target_dialect["dialect_type"] == "100":
        return (not same_del) and (not exists_del)

    return False


def augmentability_wrapper(filename, file_cells, file_content, source_dialect, unique_dialects):
    """
    This function checks for every file whether it is applicable to be augmented to the dialects in unique_dialects.
    :param filepath:
    :param unique_dialects:
    :return:
    """
    augmentability = []
    for dialect in unique_dialects:
        a = is_augmentable(file_content, file_cells, source_dialect, dialect)
        augmentability.append(a)

    return {filename: augmentability}


def insert_escape_fix_newline(file_cells: list[list[str]], delimiter='', escape='"', quote='"', percentage=0.05):
    """
    Inserts a character into a certain percentage of the cells of the dataframe at random. At least in one cell per row, it will be inserted.
    """
    out_cells = copy.deepcopy(file_cells)
    if escape or quote: #any of the two is not empty
        rrow = rng.choice(range(len(file_cells)))
        rcol = rng.choice(range(len(file_cells[rrow])))
        to_escape = (rrow, rcol)
    else:
        to_escape = (-1,-1)
    for i, row in enumerate(file_cells):
        for j, col in enumerate(row):
            str_cell = str(file_cells[i][j])
            new_cell = "".join(str_cell.splitlines())
            if rng.random() < percentage or to_escape == (i,j):
                # insert escape in a random position inside new_cell
                index = rng.integers(0, len(new_cell)) if len(new_cell) else 0
                new_cell = list(new_cell)
                if escape == '': #should not be escaped
                    new_cell.insert(index, delimiter)
                else:
                    new_cell.insert(index, quote)
                new_cell = "".join(new_cell)
            out_cells[i][j] = new_cell

    return out_cells


def augment_file(file_cells, target_filepath, target_dialectpath, source_dialect, target_dialect, source_content=None, max_rows=128):

    target_delimiter = target_dialect["delimiter"]
    target_quotechar = target_dialect["quotechar"]
    target_escapechar = target_dialect["escapechar"]

    ann_dict = {"detector": "augmented", "filename": os.path.basename(target_filepath), "dialect":
        {"delimiter": target_delimiter or "", "quotechar": target_quotechar or "", "escapechar": target_escapechar or ""}}

    quoting_scheme = csv.QUOTE_NONNUMERIC #quote minimal has a bug with cells that contains a quote
    try:
        del_none = (target_delimiter in (None, "", "n"))
        quo_none = (target_quotechar in (None, "", "n"))
        esc_none = (target_escapechar in (None, "", "n"))

        if (not esc_none) and (not source_dialect == target_dialect):
            out_cells = insert_escape_fix_newline(file_cells, target_delimiter, target_escapechar, target_quotechar)
        elif (not quo_none) and (not source_dialect == target_dialect):
            out_cells = insert_escape_fix_newline(file_cells, target_delimiter, escape='', quote=target_quotechar)
        else:
            out_cells = insert_escape_fix_newline(file_cells, target_delimiter, escape='', quote='', percentage=0)

        file_chars = set([char for row in out_cells for cell in row for char in cell] + [target_delimiter, target_quotechar, target_escapechar])
        missing_chars = list(set(chr(i) for i in range(60,1024)) - file_chars)
        fake_sep, fake_quo, fake_esc = missing_chars[0:3]  # any character not in the file set of characters
        sbuf = io.StringIO()
        # Target delimiter is None, so we only keep the first column
        if del_none:
            out_cells = [row[0:1] or [""] for row in out_cells]
            if quo_none:
                writer = csv.writer(sbuf, delimiter=fake_sep, quoting=csv.QUOTE_NONE, quotechar=fake_quo, escapechar=fake_esc)
            else:
                if esc_none:
                    writer = csv.writer(sbuf, delimiter=fake_sep, quotechar=target_quotechar,
                                        quoting=quoting_scheme, escapechar=fake_esc, doublequote=False)
                else:
                    writer = csv.writer(sbuf, delimiter=fake_sep, quotechar=target_quotechar,
                                        quoting=quoting_scheme, escapechar=target_escapechar)
        else:  # delimiter is not none
            if (not esc_none):  # target 111
                writer = csv.writer(sbuf, delimiter=target_delimiter, quotechar=target_quotechar,
                                    quoting=quoting_scheme, escapechar=target_escapechar, doublequote=False)
            else:  # target 100
                if quo_none: # target 100
                    writer = csv.writer(sbuf, delimiter=target_delimiter, quoting=csv.QUOTE_NONE, quotechar=fake_quo)
                else: # target 110
                    writer = csv.writer(sbuf, delimiter=target_delimiter, quoting=quoting_scheme, quotechar=target_quotechar)
        for row in out_cells:
            if (row == []) or row == [""]:
                sbuf.write("\n")
            else:
                writer.writerow(row)
        augmented_file = sbuf.getvalue()
        assert fake_sep not in augmented_file, "fake_sep is still in the file"
        assert fake_quo not in augmented_file, "fake_quo is still in the file"
        assert fake_esc not in augmented_file, "fake_esc is still in the file"
            
    except:
        exc = traceback.format_exc()
        print(exc, target_filepath, target_dialect)
        pdb.set_trace()
        print(row)
        return Exception(exc)

    with open(target_filepath, "w") as f:
        f.write(augmented_file)
    with open(target_dialectpath, "w") as jf:
        json.dump(ann_dict, jf)

    return None

def augment_wrapper(filename, file_cells, source_dialect, augmentable_dialects):
    for idx, dialect in enumerate(augmentable_dialects):
        target_filepath = f"{target_dir}/csv/{filename}_augmented{idx}.csv"
        target_dialectpath = f"{target_dir}/dialect/{filename}_augmented{idx}.csv_dialect.json"
        augment_file(file_cells, target_filepath, target_dialectpath, source_dialect,
                     {"delimiter": dialect[0], "quotechar": dialect[1], "escapechar": dialect[2]})
    return None


# Save space as 's'
# Save tab as 't'
# Save null as 'n'

DATA_DIR = "data/dialect_detection"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="overall")
    parser.add_argument("--max_augmented", type=int, default=100)
    parser.add_argument("--override", action="store_true")

    args = parser.parse_args()
    dataset = args.dataset
    max_augmented = int(args.max_augmented)
    override = bool(args.override)

    rng = np.random.default_rng(42)

    source_csv_dir = f"{DATA_DIR}/{dataset}/csv/"
    source_dialect_dir = f"{DATA_DIR}/{dataset}/dialect/"
    annotations_path = f"{DATA_DIR}/{dataset}/dialect_annotations.csv"
    target_dir = f"{DATA_DIR}/{dataset}_augmented/"
    target_csv_dir = f"{target_dir}/csv/"
    target_dialect_dir = f"{target_dir}/dialect/"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(target_csv_dir)
        os.makedirs(target_dialect_dir)

    if override:
        os.system(f"rm -rf {target_csv_dir}; mkdir {target_csv_dir}")
        os.system(f"rm -rf {target_dialect_dir}; mkdir {target_dialect_dir}")

        source_dialects = pqdm([source_dialect_dir+f for f in os.listdir(source_dialect_dir)], get_file_dialect, n_jobs=100)
        source_dialects_df = pd.DataFrame(source_dialects).fillna("")
        assert len(source_dialects_df[source_dialects_df["filename"] == ""]) == 0, "Missing filenames in dialect annotations"

        delimiters = set(source_dialects_df["delimiter"].unique())
        quotechars = set(source_dialects_df["quotechar"].unique())
        escapechars = set(source_dialects_df["escapechar"].unique())
        unique_dialects = [{"dialect":d+q+e, "delimiter": d, "quotechar": q, "escapechar": e,
                                    "dialect_type": str(int(bool(d))) + str(int(bool(q))) + str(int(bool(e)))}
                           for (d,q,e) in itertools.product(delimiters, quotechars, escapechars)]
        unique_dialects = [u for u in unique_dialects if not (u["escapechar"] != "" and u["quotechar"] == "")]
        unique_dialects = [u for u in unique_dialects if not ((u["delimiter"] == u["quotechar"]) and (u["delimiter"] != ""))]

        print(f"There are {len(unique_dialects)} unique dialects.")

        print("Reading source csv files..")
        files = source_dialects_df.sample(frac=1, random_state=RANDOM_SEED)["filename"].values
        file_dialects = {d.pop("filename"): d for d in source_dialects}
        args = [{"filepath": source_csv_dir + f, "dialect": file_dialects[f], "max_rows": 128} for f in files]
        file_contents = pqdm(args, parse_content, n_jobs=100, argument_type='kwargs', desc="Parsing files")
        assert len([x for x in file_contents if type(x) != dict]) == 0, "Error in parsing files"
        file_contents = {k: v for d in file_contents for k, v in d.items()}


        print("Checking augmentability of csv files..")
        args = [{"filename": f, "file_content": file_contents[f]["content"], "file_cells": file_contents[f]["file_cells"],
                 "source_dialect": file_dialects[f], "unique_dialects": unique_dialects} for f in file_contents]
        augmentability = pqdm(args, augmentability_wrapper, n_jobs=100, argument_type='kwargs', desc="Checking augmentability")
        augmentability = {k: v for d in augmentability for k, v in d.items()}
        augmentability_df = pd.DataFrame(augmentability, index=[(d["delimiter"], d["quotechar"], d["escapechar"]) for d in unique_dialects])

        for idx, (dialect, row) in enumerate(augmentability_df.iterrows()):
            indices = np.asarray(row, dtype=int)
            nonzero_indices = np.nonzero(indices)[0]
            existing_files = 0
            if existing_files >= max_augmented:
                augmentability_df.iloc[idx, :] = False
            elif len(nonzero_indices) > max_augmented-existing_files:
                drop = rng.choice(nonzero_indices, len(nonzero_indices) - (max_augmented-existing_files), replace=False)
                augmentability_df.iloc[idx, drop] = False

        print("Generating augmented csv files..")
        args = [{"filename": f, "file_cells": file_contents[f]["file_cells"],
                 "source_dialect": file_dialects[f],
                 "augmentable_dialects": [x[0] for x in augmentability_df.loc[:, f].items() if x[1]]}
                for f in file_contents]
        results = pqdm(args, augment_wrapper, n_jobs=100, argument_type='kwargs', desc="Augmenting files")
        try:
            assert len([x for x in results if x is not None]) == 0, "Some files were not augmented"
        except AssertionError as e:
            pdb.set_trace()
        augmentability_df.to_csv(f"{target_dir}/augmentability.csv", index=False)


        print("Pruning augmented csv files..")
        list_augmented = pqdm([target_dialect_dir+f for f in os.listdir(target_dialect_dir)], get_file_dialect, n_jobs=100)
        ddf = pd.DataFrame(list_augmented)

        dialect_counts = dict(ddf.groupby(["delimiter", "quotechar", "escapechar"]).count()["filename"].items())

        to_prune = [(x[0], x[1] - max_augmented) for x in list(ddf.groupby("dialect").count()["filename"].items())
                    if x[1] > max_augmented]
        for dialect, num in to_prune:
            dialect_files = ddf[ddf["dialect"] == dialect]["filename"].values
            dialect_files = rng.choice(dialect_files, num, replace=False)
            for f in dialect_files:
                os.remove(f"{target_dir}csv/{f}")
                os.remove(f"{target_dir}dialect/{f}_dialect.json")

    list_augmented = pqdm([target_dialect_dir+f for f in os.listdir(target_dialect_dir)], get_file_dialect, n_jobs=100)
    ddf = pd.DataFrame(list_augmented)
    print(ddf.groupby("dialect").count().sort_values(by="filename", ascending=False))

    print("Create annotations csv..")
    dialect_annotations_file = f"{target_dir}/dialect_annotations.csv"
    ddf.to_csv(dialect_annotations_file, index=False, quoting=csv.QUOTE_ALL)


