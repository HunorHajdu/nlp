# BERT vs a Custom Modell for mask filling tasks

This project uses BERT and a custom made model for some mask filling task

# Usage

Clone this repo then install the requirments by running:

```py
pip install -r requirements.txt
```

# Dataset

Download the dataset from [here](https://drive.google.com/drive/folders/1mUiiYJsfHgI3y-LxhoqigmunxCBhT8Yw), then put them into a a folder called datasets.

# Runing the models

You can run the models by executing the following commands:
- for the BERT based model
```
python src/main.py
```
- for our custom model
```
python src/simple-model.py
```