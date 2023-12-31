import json
import os
import numpy as np
from tqdm import tqdm
import re
import pdb
import itertools
from pqdm.processes import pqdm
from Levenshtein import distance as edit_distance

N_JOBS = 100

def containenglish(string): # check if the string contains any English words
    return bool(re.search('[a-zA-Z]', string))

def match_quantity(datestr): 
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    events_dict = ['Prix','Olympics','Championships', 'Open', 'Challenger', 'Trophy', 'Tournament']
    flag = 0
    for char in datestr:
        if char in numbers:
            flag = 1
            break
    if flag == 0:
        data_type = 'WORK_OF_ART'
    else:
        flag2 = 0
        for word in datestr.split(' '):
            if word in events_dict:
                flag2 = 1
                break
        if flag2 == 1:
            data_type = 'EVENT'
        else:
            data_type = 'QUANTITY'
    return data_type

def match_date(datestr): # Divide the DATE
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    events_dict = ['Prix','Olympics','Championships', 'Open', 'Challenger', 'Trophy', 'Tournament']
    if datestr.isdigit(): #YYYY
        data_type = 'DATE1'
    else:
        if containenglish(datestr): #Jan
            flag = 0
            for char in datestr:
                if char in numbers:
                    flag = 1
                    break
            if flag == 0:
                data_type = 'WORK_OF_ART'    
            else:
                flag2 = 0
                for word in datestr.split(' '):
                    if word in events_dict:
                        flag2 = 1
                        break
                if flag2 == 1:
                    data_type = 'EVENT'
                else:
                    data_type = 'DATE2'
        else:
            splitted = datestr.split('-')
            if len(splitted) == 3: #YYYY-MM-DD
                data_type = 'DATE3'
            elif len(splitted) == 2: #MM-DD
                data_type = 'DATE4'
            else:
                data_type = 'DATE5'
    return data_type

def match_name(namestr): #Divide PERSON
    if '.' in namestr:
        data_type = 'PERSON1'
    else:
        data_type = 'PERSON2'
    return data_type


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', default="clean" ,help='Specify the type argument')
args = parser.parse_args()
type_arg = args.type
data_dir = f"../data_{type_arg}"

ALL_DATASETS = ["cius","deex","govuk","mendeley","saus", "troy"]
json_summary_base = f'{data_dir}/json/'
out_base = f'{data_dir}/json/'

def preprocess_date_name(): # Process PERSON and DATE
    
    print('Now process the dates, quantities and names format')
    for dataset in ALL_DATASETS:
        json_summary_dir = json_summary_base+str(dataset)
        out_dir = out_base+str(dataset)
        for json_file_name in tqdm(os.listdir(json_summary_dir)):
            json_summary_path = json_summary_dir+'/'+json_file_name
            out_path = out_dir+'/'+json_file_name
            try:
                with open(json_summary_path, 'r') as jf:
                    content = json.load(jf)
            except:
                pdb.set_trace()
            col_types = content['most_common']
            table_content = content['content']
            out_json = content
            for i, data_type in enumerate(col_types):
                if data_type == 'DATE':
                    candidate_str = np.array(content['content'])[0, i]
                    if not candidate_str:
                        candidate_str = np.array(content['content'])[1, i]
                    out_json['most_common'][i] = match_date(candidate_str)
                if data_type == 'PERSON':
                    candidate_str = np.array(content['content'])[0, i]
                    if not candidate_str:
                        candidate_str = np.array(content['content'])[1, i]
                    out_json['most_common'][i] = match_name(candidate_str)
                if data_type == 'QUANTITY':
                    candidate_str = np.array(content['content'])[0, i]
                    if not candidate_str:
                        candidate_str = np.array(content['content'])[1, i]
                    out_json['most_common'][i] = match_quantity(candidate_str)
            with open(out_path, 'w') as of:
                json.dump(out_json, of)

def find_related(): # Align related tables

    pattern_dict = {}
    print('Now finding related tables')
    for dataset in ALL_DATASETS:
        json_path = json_summary_base+str(dataset)
        for json_file in os.listdir(json_path):
            json_file_path = json_path+'/'+json_file
            with open(json_file_path, 'r') as load_f:
                load_dict = json.load(load_f)
            most_common_key = str(load_dict['most_common'])
            json_id = str(dataset)+'-'+json_file
            if most_common_key in pattern_dict.keys():
                pattern_dict[most_common_key].append(json_id)
            else:
                pattern_dict[most_common_key] = [json_id]
    for key_pattern in pattern_dict.keys():
        if len(pattern_dict[key_pattern]) == 1:
            file_name = pattern_dict[key_pattern][0]
            dataset, file_id = file_name.split('-')
            file_path = json_summary_base+dataset+'/'+file_id
            
            with open(file_path, 'r') as load_file:
                load_dict = json.load(load_file)
            out_path = out_base+dataset+'/'+file_id
            related_table = [load_dict['filename']]
            load_dict['related_table'] = related_table
            with open(out_path, 'w') as dump:
                json.dump(load_dict, dump)
        else:
            table_ids = []
            for file_name in pattern_dict[key_pattern]:
                dataset, file_id = file_name.split('-')
                file_path = json_summary_base+dataset+'/'+file_id
                with open(file_path, 'r') as load_file:
                    load_dict = json.load(load_file)
                if load_dict['filename'] in table_ids:
                    continue
                else:
                    table_ids.append(load_dict['filename'])
            if len(table_ids) == 1:
                for file_name in pattern_dict[key_pattern]:
                    dataset, file_id = file_name.split('-')
                    file_path = json_summary_base+dataset+'/'+file_id
                    with open(file_path, 'r') as load_file:
                        load_dict = json.load(load_file)
                    load_dict['related_table'] = [load_dict['filename']]
                    out_path = out_base+dataset+'/'+file_id
                    with open(out_path, 'w') as dump:
                        json.dump(load_dict, dump)
            else:
                for file_name in pattern_dict[key_pattern]:
                    dataset, file_id = file_name.split('-')
                    file_path = json_summary_base+dataset+'/'+file_id
                    with open(file_path, 'r') as load_file:
                        load_dict = json.load(load_file)
                    current_table_id = [load_dict['filename']]
                    for item in table_ids:
                        if not item in current_table_id:
                            current_table_id.append(item)
                    out_path = out_base+dataset+'/'+file_id
                    out_dict = load_dict
                    out_dict['related_table'] = current_table_id
                    with open(out_path, 'w') as dump:
                        json.dump(out_dict, dump)

def editDistance(str1, str2):
 
    if str1 == str2:
        return 0

    m = len(str1)
    n = len(str2)
    
    # Our add: if length of the string is too long, 
    # return more than sqrt(len(str2))
    if abs(m-n) > (n)**0.5:
        return (n**0.5)+1

    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n
 
    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m
 
    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    if str1[m-1] == str2[n-1]:
        return editDistance(str1, str2, m-1, n-1)
 
    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    return 1 + min(editDistance(str1, str2, m, n-1),    # Insert
                   editDistance(str1, str2, m-1, n),    # Remove
                   editDistance(str1, str2, m-1, n-1)    # Replace
                   )

def parallel_edit_distance(combination):
    table_1, table_2, tstr_1, tstr_2 = combination
    # return (table_1,table_2, editDistance(tstr_1, tstr_2))
    return (table_1,table_2, edit_distance(tstr_1, tstr_2))


def compute_edit_distance(): # Compute edit distance
    output_dir = f'{data_dir}/distance-file/'
    type_dict = {'CARDINAL':'A', 'DATE1':'B','DATE2':'C', 'DATE3':'D', 'DATE4':'E', 'DATE5':'F', 'EVENT':'G', 'FAC':'H', 'GPE':'I', 'LANGUAGE':'J', 'LAW':'K', 'LOC':'L', 'MONEY':'M', 'NORP':'N', 'ORDINAL':'O', 'ORG':'P', 'PERCENT':'Q', 'PERSON1':'R','PERSON2':'S', 'PRODUCT':'T', 'QUANTITY':'U', 'TIME':'V', 'WORK_OF_ART':'W', 'EMPTY':'X'}
    table_strings = {}
    print('Now computing edit distances')
    for dataset in ALL_DATASETS:
        json_summary_dir = json_summary_base+str(dataset)
        for json_file_name in tqdm(os.listdir(json_summary_dir)):
            json_summary_path = json_summary_dir+'/'+json_file_name
            with open(json_summary_path, 'r') as jf:
                content = json.load(jf)
            col_types = content['most_common']
            table_name = content['filename']
            table_string = ''
            for col in col_types:
                table_string += type_dict[col]
            table_strings[table_name] = table_string

    edit_distances = {}
    combinations = itertools.combinations(table_strings.keys(), 2)
    args = ((table_1, table_2, table_strings[table_1], table_strings[table_2]) for table_1, table_2 in combinations)
    results = pqdm(args, parallel_edit_distance, n_jobs=N_JOBS)
    edit_distances = {(table_1, table_2): distance for table_1, table_2, distance in results}

    for table_2 in tqdm(table_strings):
        distances = {}
        for table_1 in table_strings:
            tstr_1  = table_strings[table_1]
            tstr_2 = table_strings[table_2]
            if tstr_1 == tstr_2:
                distances['0'] = distances.get('0',[]) + [table_1]
                distances['0-type'] = distances.get('0-type',[]) + [tstr_1]
            else:
                try:
                    distance = edit_distances[(table_1, table_2)]
                except KeyError:
                    distance = edit_distances[(table_2, table_1)]
                if distance > (len(tstr_2))**0.5:
                    continue
                distances[str(distance)] = distances.get(str(distance),[]) + [table_1]
                distances[str(distance)+'-type'] = distances.get(str(distance)+'-type',[]) + [tstr_1]
        with open(output_dir+table_2[:-3]+'json', 'w') as of:
            json.dump(distances, of)

def find_sub_related(): # Align sub related tables
    print('Now adding sub-related tables')
    distance_dir = f'{data_dir}/distance-file/'
    if not os.path.exists(distance_dir):
        os.makedirs(distance_dir)

    for dataset in ALL_DATASETS:
        json_summary_dir = json_summary_base+str(dataset)+'/'
        for json_filename in tqdm(os.listdir(json_summary_dir)):
            json_summary_path = json_summary_dir + json_filename
            with open(json_summary_path) as of:
                data = json.load(of)
            json_file = data['filename'][:-4]+'.json'
            distance_path = distance_dir+json_file
            data['subtable'] = []
            data['subtable-type'] = []
            with open(distance_path) as df:
                distances = json.load(df)
            for key in distances.keys():
                if len(key.split("-")) == 2:
                    if key == "0-type":
                        data['table-type'] = distances[key][0]
                    continue
                elif key == '0':
                    continue
                else:
                    for index in range(len(distances[key])):
                        data['subtable'].append(distances[key][index])
                        data['subtable-type'].append(distances[key+'-type'][index])
            with open(json_summary_path, 'w') as f:
                json.dump(data, f)

if __name__ == '__main__':
    preprocess_date_name()
    find_related()
    compute_edit_distance()
    find_sub_related()