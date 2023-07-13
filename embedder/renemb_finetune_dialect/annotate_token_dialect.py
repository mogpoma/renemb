# this scripts annotates the tokens of a given file with their dialect - this is used for the finetuning of the dialect model
# the output is a txt file with the same name as the input file, but with the suffix _dialect.txt
# the output file contains the annotations of the input file tokens, line by line, separated by space
# the annotations are the following:
# C - the token is part of a cell
# D - the token represents a delimiter
# Q - the token represents a quotechar
# E - the token represents an escapechar
import argparse
# there are three cases:
# Case 1: dialect has a delimiter, a quotechar but NOT an escapechar
# in this case the tokens are annotated as follows:
# S1: input: any, output: C, state S1
#     input: delimiter, output: D, state S1
#     input: quotechar, output: Q, goto state S2
# S2: input: any, output: C, state S2
#     input: delimiter, output: C, state S2
#     input: quotechar, output: Q, goto state S1

# Case 2: dialect has a delimiter, a quotechar and an escapechar different from the quotechar
# in this case the tokens are annotated as follows:
# S1: input: any, output: C, state S1
#     input: delimiter, output: D, state S1
#     input: quotechar, output: Q, goto state S2
#     input: escapechar, raise Error
# S2: input: any, output: C, state S2
#     input: delimiter, output: C, state S2
#     input: quotechar, output: Q, goto state S1
#     input: escapechar, output: E, goto state S3
# S3: input: any, raise Error
#     input: delimiter, raise Error
#     input: quotechar, output: C, goto state S2
#     input: escapechar raise Error

# Case 3: dialect has a delimiter, a quotechar and an escapechar equal to the quotechar
# in this case the tokens are annotated as follows:
# S1: input: any, output: C, state S1
#     input: delimiter, output: D, state S1
#     input: quotescape, output: Q, goto state S2
# S2: input: any, output: C, state S2
#     input: delimiter, output: C, state S2
#     input: quotescape, output: None, goto state S3
# S3: input: any, raise Error
#     input: delimiter, output: Q D, goto state S1
#     input: quotescape, output: E C, goto state S2


import sys

import chardet

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import os
import json
from embedder.pattern_tokenizer import PatternTokenizer
from pqdm.processes import pqdm


class AnnotationTags():
    CELL = "C"
    DEL = "D"
    QUOTE = "Q"
    ESCAPE = "E"
    

class DialectAnnotator():

    def __init__(self, tokenizer: PatternTokenizer,
                 delimiter: str,
                quotechar:str ,
                escapechar: str,
                 max_rows: int = None,
                 *args, **kwargs
                 ):
        self.tokenizer = tokenizer
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.escapechar = escapechar
        self.max_rows = max_rows

    def annotate_no_escape(self,filename, line_list):
        tag_list = []
        state = 0
        for line in line_list:
            tag_row = []
            for c in line:
                if state == 0:
                    if c == self.delimiter:
                        tag_row.append(AnnotationTags.DEL)
                    elif c == self.quotechar:
                        tag_row.append(AnnotationTags.QUOTE)
                        state = 1
                    else:
                        tag_row.append(AnnotationTags.CELL)
                elif state == 1:
                    if c == self.quotechar:
                        tag_row.append(AnnotationTags.QUOTE)
                        state = 0
                    else:
                        tag_row.append(AnnotationTags.CELL)
            tag_list.append(tag_row)

        assert state == 0, f"{filename}: the quotechar is not closed"
        return tag_list

    def annotate_different_escape(self,filename, line_list, text_lines=None):
        tag_list = []
        state = 0
        for idx, line in enumerate(line_list):
            tag_row = []
            for c in line:
                if state == 0:
                    if c == self.delimiter:
                        tag_row.append(AnnotationTags.DEL)
                    elif c == self.quotechar:
                        tag_row.append(AnnotationTags.QUOTE)
                        state = 1
                    elif c == self.escapechar:
                        # pdb.set_trace()
                        print(f"{filename}: Escapechar cannot be in unquoted field")
                    else:
                        tag_row.append(AnnotationTags.CELL)
                elif state == 1:
                    if c == self.quotechar:
                        tag_row.append(AnnotationTags.QUOTE)
                        state = 0
                    elif c == self.escapechar:
                        tag_row.append(AnnotationTags.ESCAPE)
                        state = 2
                    else:
                        tag_row.append(AnnotationTags.CELL)
                elif state == 2:
                    if c == self.quotechar:
                        tag_row.append(AnnotationTags.CELL)
                        state = 1
                    else:
                        tag_row += [AnnotationTags.CELL, AnnotationTags.CELL]
                        print(f"{filename} : Escape is being used without a quotation mark afterwards")
                        state = 1

            tag_list.append(tag_row)

        assert state == 0, f"{filename} : the quotechar is not closed"
        return tag_list

    def annotate_quotescape(self,filename, line_list, text_lines=None):
        tag_list = []
        state = 0
        for line_idx, line in enumerate(line_list):
            tag_row = []
            for idx,c in enumerate(line):
                if state == 0:
                    if c == self.delimiter:
                        tag_row.append(AnnotationTags.DEL)
                    elif c == self.quotechar:
                        tag_row.append(AnnotationTags.QUOTE)
                        state = 1
                    else:
                        tag_row.append(AnnotationTags.CELL)
                elif state == 1:
                    if c == self.quotechar and idx < len(line)-1:
                        state = 2
                    elif c == self.quotechar and idx == len(line)-1:
                        tag_row.append(AnnotationTags.QUOTE)
                        state = 0
                    else:
                        tag_row.append(AnnotationTags.CELL)
                elif state == 2:
                    if c == self.delimiter:
                        tag_row += [AnnotationTags.QUOTE, AnnotationTags.DEL]
                        state = 0
                    elif c == self.quotechar:
                        tag_row += [AnnotationTags.ESCAPE, AnnotationTags.CELL]
                        state = 1
                    elif c.strip() == "" and idx == len(line):
                        tag_row += [AnnotationTags.QUOTE, AnnotationTags.CELL]
                        state = 0
                        print(f"{filename} : Cell is being closed with an extra space afterwards: ")
                    else:
                        tag_row += [AnnotationTags.CELL, AnnotationTags.CELL]
                        print(f"{filename} : Escape is being used without a quotation mark afterwards")
                        state=1
            tag_list.append(tag_row)

        assert state == 0, "The quotechar is not closed"
        return tag_list
    
    def annotate_file(self, input_file, output_file):
        tokenizer = PatternTokenizer()
        # read the input file
        try:
            with open(input_file, 'r') as f:
                data = f.read()
        except UnicodeDecodeError:
            rawfile = open(f"{input_file}", "rb").read()
            encoding = chardet.detect(rawfile)["encoding"]
            data = rawfile.decode(encoding)
        lines = data.splitlines()[:self.max_rows]
        # annotate the tokens
        tokenized_lines = list(map(lambda x: tokenizer(x, return_text=True), lines))

        if self.escapechar is None:
            tag_list = self.annotate_no_escape(os.path.basename(input_file),tokenized_lines)
        elif self.escapechar == self.quotechar:
            tag_list = self.annotate_quotescape(os.path.basename(input_file),tokenized_lines, lines)
        else:
            tag_list = self.annotate_different_escape(os.path.basename(input_file), tokenized_lines, lines)
        # write the output file
        tag_str = "\n".join(map(lambda x: " ".join(x), tag_list))
        if tag_list[-1] == [""]:
            tag_str += "\n"
        assert "CQQ" not in tag_str, "The quotechar is not closed"
        assert "QQC" not in tag_str, "The quotechar is not closed"

        with open(output_file, 'w') as f:
            f.write(tag_str)
        return None


# Take the first file from data path "data/annotated" and annotate it
# read the annotations from data path "/dialect/{filename}_dialect.json"
# and write the annotated file to data path "/dialect_tags/{filename}_tags.csv"

def annotate(file, override):
    filepath = f"{data_path}/csv/{file}"
    dialect_path = f"{data_path}/dialect/{file}_dialect.json"
    outpath = f"{data_path}/dialect_tags/{file}_tags.csv"

    if os.path.exists(outpath) and not override:
        rawfile = open(f"{filepath}", "rb").read()
        try:
            f = rawfile.decode("utf-8")
        except UnicodeDecodeError:
            encoding = chardet.detect(rawfile)["encoding"]
            f = rawfile.decode(encoding)
        rows = f.splitlines()
        rows = [r for r in rows][:128]
        tag_file = open(outpath, "r").read()
        tag_rows = tag_file.splitlines()[:128]

        for idx,r in enumerate(rows):
            if len(r) and (len(PatternTokenizer()(r)) != len(tag_rows[idx].split(" "))):
                print("Found error in file ", file)
                override = True
                break
        if len(tag_rows) != len(rows):
            override = True
    
    if not override:
        return
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    annotation = json.load(open(dialect_path))
    dialect_annotator = DialectAnnotator(PatternTokenizer(),**annotation["dialect"],max_rows=None)
    try:
        dialect_annotator.annotate_file(filepath,outpath)
    except Exception as e:
        print(e)
        return "File "+file+" : "+str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="overall_augmented")
    parser.add_argument("--override", action='store_true')

    dataset = parser.parse_args().dataset
    override = bool(parser.parse_args().override)

    data_path = f"data/dialect_detection/{dataset}/"
    args = [{"file":f, "override":override} for f in os.listdir(data_path+"csv")]
    results = pqdm(args, annotate, n_jobs=100, desc="Annotating files", argument_type='kwargs')


    errors = [str(r) for r in results if r is not None]
    with open(data_path+"tag_errors.txt", "w") as f:
        f.write("\n".join(errors))
    print("Number of error files: ", len(errors))
