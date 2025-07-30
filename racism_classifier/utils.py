import pandas as pd
from datasets import Dataset
from racism_classifier.config import DATA_PATH, MODEL_DIR_PATH, BERT_MODEL_NAME
from transformers import AutoTokenizer
from huggingface_hub import ModelCardData, ModelCard
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from transformers import Trainer

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

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    This loss is designed to address class imbalance by down-weighting the loss contribution.
    Args:
    gamma: weight for "easy" examples. If 0, gamma=cross-entropy (default 2)
    higher gamma= more focus on hard to classify examples.
    alpha: tensor with shape [num_classes] or scalar for pos v.s. neg class binary.
    higher alpha=loss from class amplified
    Use for underrepresented classes.
    reduction: 'none', 'mean', or 'sum'. Default is 'mean'.
    None is used so that manual weighting for each loss can be done
    After Loss is calculated, it must be reduced to mean or sum.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLossTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, alpha=None,reduction="mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha,reduction=reduction)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): 
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
