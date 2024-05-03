#!/bin/bash

# Array of directories to process
directories+=("archive1/Actor_07/" "archive1/Actor_08/" "archive1/Actor_09/" "archive1/Actor_10/" "archive1/Actor_11/" "archive1/Actor_12/" "archive1/Actor_13/" "archive1/Actor_14/" "archive1/Actor_15/" "archive1/Actor_16/" "archive1/Actor_17/" "archive1/Actor_18/" "archive1/Actor_19/" "archive1/Actor_20/" "archive1/Actor_21/" "archive1/Actor_22/" "archive1/Actor_23/" "archive1/Actor_24/")

# Iterate over each directory
for dir in "${directories[@]}"
do
    echo "Processing directory: $dir"
    python extract_features.py "$dir"
done
