@echo off
python -m venv char_class_env
call char_class_env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

echo Please provide the full path to the dataset:
set /p dataset_path=

python main.py "%dataset_path%"

cd output_files
start complete_results.html
pause