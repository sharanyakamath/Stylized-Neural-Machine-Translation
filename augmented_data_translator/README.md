The model we made is available here: https://drive.google.com/drive/folders/1NyTTTMZEVzC0TpyXaB1LhNWnZ1AjJ-ml?usp=sharing. It essentially starts with T5-small and is further fine-tuned.

1. For training, use the script t5_translator.py by changing path your dataset in data_utils.py.
2. For translation, use the script translate_all.py with the dataset of your interest. Change input path, output path in translate_all.py.
3. All data formats can be found in data directory.
