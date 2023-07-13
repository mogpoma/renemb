#!/bin/bash
type="autoclean"

python 0_transform_to_json.py --type $type
python 1_NER_extraction.py --type $type
python 2_pre-process.py --type $type
python 3_make_json_input.py --type $type
python 4_jaccard_filterjson.py --type $type
python 5_split-datasets.py --type $type --n_folds 10