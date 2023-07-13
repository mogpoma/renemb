import csv
import json
import os

def read_csv(csv_dir):
    csv_lines = csv.reader(open(csv_dir, "r", encoding="utf-8"))
    content = []
    num_row = 0
    for i, line in enumerate(csv_lines):
        if i == 0:
            header = line
            num_col = len(header)
            
        else:
            content.append(line)
            num_row += 1
    num_cell = num_col*num_row
    return header, content, num_col, num_row, num_cell


ALL_DATASETS = ["cius","deex","govuk","mendeley","saus", "troy"]


# add a command line argument to specify the dataset with name "dataset"
# dataset = args.dataset

# Create the ArgumentParser object
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--type', default="clean" ,help='Specify the type argument')
args = parser.parse_args()
type_arg = args.type
data_dir = f"../data_{type_arg}"

for dataset in ALL_DATASETS:
    csv_reader = csv.reader(open(f"{data_dir}/raw_data/"+dataset+"_gt.csv"))
    #gt is ground_truth file
    csv_dir = f"{data_dir}/raw_data/"+str(dataset)+"/"
    output_dir = f"{data_dir}/json/"+str(dataset)+"/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i,line in enumerate(csv_reader):
        file_name = line[0]
        target_col = line[2]
        label = line[3]
        file_dir = csv_dir+file_name
        if not os.path.exists(file_dir):
            print(file_dir + " not exists")
        header, content, num_col, num_row, num_cell = read_csv(file_dir)
        dict = {}
        dict['filename'] = file_name
        dict['header'] = header
        dict['content'] = content
        dict['target'] = target_col
        dict['label'] = label
        output_path = output_dir+str(i)+'.json'
        with open(output_path, "w") as outfile:
            json.dump(dict, outfile)