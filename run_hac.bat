@echo off
python -m venv char_class_env
call char_class_env\Scripts\activate
pip install -r requirements.txt
python main.py
pause