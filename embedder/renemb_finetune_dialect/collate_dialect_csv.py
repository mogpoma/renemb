""" This script collates the dialect .json files into a unique .csv file.
"""
import os
import json
import pandas as pd
from pqdm.processes import pqdm
import csv
import pdb

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
        dialect_type = "???"
    if len(dialect) > 3:
        print(dialect_path[:-13] + " has a dialect with more than 3 characters")
        print(f"\t |{delimiter}| |{quotechar}| |{escapechar}|")
        return {"filename": os.path.basename(dialect_path[:-13]), 
                "dialect": dialect, 
                "delimiter": dialect_dict["delimiter"][:1], 
                "quotechar": dialect_dict["quotechar"][:1],
                "escapechar": dialect_dict["escapechar"][:1], 
                "dialect_type": dialect_type}
    return {"filename": os.path.basename(dialect_path[:-13]), "dialect": dialect, "delimiter": dialect_dict["delimiter"], "quotechar": dialect_dict["quotechar"],
                  "escapechar": dialect_dict["escapechar"], "dialect_type": dialect_type}


source_dialect_dir = "data/dialect_detection/overall/dialect/"
target_dir = "data/dialect_detection/overall/"

source_dialects = pqdm([source_dialect_dir+f for f in os.listdir(source_dialect_dir)], get_file_dialect, n_jobs=100)

if len([x for x in source_dialects if type(x)!=dict]):
    pdb.set_trace()
source_dialects_df = pd.DataFrame(source_dialects).fillna("").set_index("filename")
source_dialects_df.to_csv(os.path.join(target_dir, "dialect_annotations.csv"), quoting=csv.QUOTE_ALL)
