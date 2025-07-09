import pandas as pd
from datasets import Dataset
from racism_classifier.config import DATA_PATH, MODEL_DIR_PATH, BERT_MODEL_NAME
from transformers import AutoTokenizer
from huggingface_hub import ModelCardData, ModelCard
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

# Model cart

def create_model_card():
    card_data = ModelCardData(language='en',
                              base_model=BERT_MODEL_NAME,
                              datasets="Dataset from Rainer"
                              )

    card = ModelCard.from_template(
        card_data,
        model_id='my-cool-model',
        model_description="this model does this and that",
        developers="Nate Raw",
        repo="https://github.com/huggingface/huggingface_hub",
    )
    card.save(f'{MODEL_DIR_PATH}/README.md')
    print(card)