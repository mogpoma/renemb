#!/bin/bash
type="unprepared"
echo $type
python RECA-train+test.py --type $type --n_folds 10
python RECA-test-from-pre-trained.py --type $type --n_folds 10

type="autoclean"
echo $type
python RECA-train+test.py --type $type --n_folds 10
python RECA-test-from-pre-trained.py --type $type --n_folds 10

type="clean"
echo $type
python RECA-train+test.py --type $type --n_folds 10
python RECA-test-from-pre-trained.py --type $type --n_folds 10