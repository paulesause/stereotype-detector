import pandas as pd
from datasets import Dataset
from racism_classifier.config import DATA_PATH
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

def load_data(file_path:str=DATA_PATH):
    """Loads the data from an xls file and returns a Dataset object"""
    df = pd.read_excel(file_path)
    return Dataset.from_pandas(df)


# Load tokenizer 
_tokenizer = None

def get_tokenizer_from_huggingface():
    global _tokenizer
    if _tokenizer is None:
        hugging_face_repro = get_huggingface_repro()
        _tokenizer = AutoTokenizer.from_pretrained(hugging_face_repro)
    return _tokenizer

# Load huggingface repro name
_hugging_face_repro = None

def get_huggingface_repro():
    global _hugging_face_repro
    if _hugging_face_repro == None:
        _hugging_face_repro = os.getenv("HUGGING_FACE_BERT_MODEL_REPRO")
    return _hugging_face_repro

# Load huggingface access token
_hugging_face_token = None

def get_huggingface_token():
    global _hugging_face_token
    if _hugging_face_token == None:
        _hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
    return _hugging_face_token

