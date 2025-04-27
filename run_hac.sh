#!/bin/bash

python3 -m venv char_class_env
source char_class_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Please provide the full path to the dataset:"
read dataset_path

python main.py "$dataset_path"

xdg-open output_files/complete_results.html
read -p "Press any key to exit..."