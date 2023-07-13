import spacy
import os
import json
import numpy as np
from collections import Counter
from tqdm import tqdm
from pqdm.processes import pqdm

N_JOBS = 100


# Create the ArgumentParser object
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', default="clean" ,help='Specify the type argument')
args = parser.parse_args()
type_arg = args.type
data_dir = f"../data_{type_arg}"

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')
table_dir = f'{data_dir}/json/'
out_dir = f'{data_dir}/json/'
ALL_DATASETS = ["cius","deex","govuk","mendeley","saus", "troy"]
  
def process_json(json_file):
    json_file_path = json_dir+'/'+json_file
    out_json_file_path = out_json_dir+'/'+json_file
    with open(json_file_path, 'r') as load_f:
        load_dict = json.load(load_f)

    table_content = np.array(load_dict['content'])
    width = len(load_dict['header'])
    col_ne = []
    most_common = []
    for col_index in range(width):
        column_string = ''
        cur_ne = []
        for cell in table_content[:,col_index]:
            column_string += cell
            column_string += ' ; '
        doc = nlp(column_string)
        for ent in doc.ents:
            cur_ne.append(ent.label_)
        col_ne.append(cur_ne)
        if cur_ne:
            most_common.append(Counter(cur_ne).most_common(1)[0][0])
        else:
            most_common.append('EMPTY')
    load_dict['table_NE'] = col_ne
    load_dict['most_common'] = most_common
    with open(out_json_file_path, 'w') as out_f:
        json.dump(load_dict, out_f)

for dataset in ALL_DATASETS:
    json_dir = table_dir+str(dataset)
    out_json_dir = out_dir+str(dataset)
    for json_file in tqdm(os.listdir(json_dir)):
        process_json(json_file)
    # pqdm(os.listdir(json_dir), process_json, n_jobs=N_JOBS)

        # print(json_file)
    
