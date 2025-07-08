from transformers import AutoModelForSequenceClassification
from racism_classifier.config import BERT_MODEL_NAME, NUMBER_OF_LABELS, ID2LABEL_MAP, LABEL2ID_MAP

def model_init_BERT():
    return AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, 
        num_labels=NUMBER_OF_LABELS, 
        id2label=ID2LABEL_MAP, 
        label2id=LABEL2ID_MAP
    )


def optuna_hp_space_BERT(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1,2,3,4]),
    }

def compute_objective_BERT(metrics):
    return metrics["eval_f1_macro"]
