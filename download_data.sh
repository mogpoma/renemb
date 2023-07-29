#!/bin/bash
pushd data/gittables
chmod +x download.sh
./download.sh
popd
pushd dialect_detection
chmod +x extract.sh
./extract.sh
popd
pushd row_classification
chmod +x extract.sh
./extract.sh
popd

