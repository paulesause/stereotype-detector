import pandas as pd
from datasets import Dataset
from stereotype_detector.config import DATA_PATH, MODEL_DIR_PATH
from transformers import Trainer, TrainingArguments
from huggingface_hub import ModelCardData, ModelCard
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field

load_dotenv()


def load_data(file_path: str = DATA_PATH):
    """Loads the data from an xls file and returns a Dataset object"""
    df = pd.read_excel(file_path)
    return Dataset.from_pandas(df)


# Load huggingface user name
_hugging_face_user_name = None

def get_huggingface_user_name():
    global _hugging_face_user_name
    if _hugging_face_user_name == None:
        _hugging_face_repro = os.getenv("HUGGING_FACE_USER_NAME")
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
    card_data = ModelCardData(
        language="en", base_model="base_model", datasets="Dataset from Rainer"
    )

    card = ModelCard.from_template(
        card_data,
        model_id="my-cool-model",
        model_description="this model does this and that",
        developers="Nate Raw",
        repo="https://github.com/huggingface/huggingface_hub",
    )
    card.save(f"{MODEL_DIR_PATH}/README.md")
    print(card)


def _get_backbone_and_layers(model):
    """
    Return (embeddings_module, list_of_layer_modules) for supported models.
    Supports:
      - DistilBERT: model.distilbert.embeddings, model.distilbert.transformer.layer (len=6)
      - BERT/RoBERTa-style: model.bert.embeddings, model.bert.encoder.layer (len=12 for base)
    """
    # DistilBERT
    if hasattr(model, "distilbert"):
        embeddings = model.distilbert.embeddings
        layers = list(model.distilbert.transformer.layer)
        return embeddings, layers, "distilbert"

    # BERT (incl. multilingual, GBERT)
    if hasattr(model, "bert"):
        embeddings = model.bert.embeddings
        layers = list(model.bert.encoder.layer)
        return embeddings, layers, "bert"

    raise ValueError(
        "Unsupported model architecture for layer freezing. "
        "Expected one of: distilbert, bert, roberta."
    )


# freeze layers (for Distilbert)
def freeze_layers(
    model, freeze_embeddings: bool = False, num_transformer_layers_freeze: int = 0
):
    """
    Freezes layers of a DistilBERT model during fine-tuning.

    Parameters:
    - model: The DistilBERT model (e.g., DistilBertForSequenceClassification).
    - freeze_embeddings (bool): If True, the embedding layer will be frozen (no gradient updates).
    - num_transformer_layers_freeze (int): Number of bottom transformer layers to freeze (0–6 for DistilBERT).

    This function modifies the model in-place by disabling gradients for the specified layers.
    """

    embeddings, layers, arch = _get_backbone_and_layers(model)

    if freeze_embeddings:
        for param in embeddings.parameters():
            param.requires_grad = False
        print("Embedding layer frozen.")
    else:
        print("Embedding layer not frozen.")

    for i, layer in enumerate(layers):
        freeze = i < num_transformer_layers_freeze
        for param in layer.parameters():
            param.requires_grad = not freeze
        print(f"Transformer layer {i} {'frozen' if freeze else 'not frozen'}.")


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

    def __init__(self, gamma=2.0, alpha=None, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert alpha to tensor if it's a float (for multi-class, all classes get alpha)
        weight = self.alpha
        if isinstance(self.alpha, float):
            num_classes = logits.size(-1)
            # You can customize this: here, all classes get the same weight
            weight = torch.full(
                (num_classes,), self.alpha, device=logits.device, dtype=logits.dtype
            )
        elif isinstance(self.alpha, list):
            weight = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
        # else: if None or already a tensor, use as is

        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", weight=weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, alpha=None, reduction="mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.focal_loss(logits, labels)
        print("Using FocalLoss")
        return (loss, outputs) if return_outputs else loss


@dataclass
class CustomTrainingArguments(TrainingArguments):
    alpha: float = field(default=None, metadata={"help": "Alpha for focal loss"})
    gamma: float = field(default=2.0, metadata={"help": "Gamma for focal loss"})
