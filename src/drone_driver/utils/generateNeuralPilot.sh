#!/bin/bash

# Gets the arguments
output_dir=$1
model_path_1=$2

if [ -z "$output_dir" ] || [ -z "$model_path_1" ]; then
    echo "Usage: $0 <output_dir> <model_name>"
    exit 1
fi


# Generates the dataset 
./generateDataset.sh 270 $output_dir


# Training
python3 ../src/train.py --dataset_path $output_dir --network_path $model_path_1 #--weights True


./trainAndCheck.sh $output_dir $output_dir $model_path_1 false