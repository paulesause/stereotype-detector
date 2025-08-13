"""
This script finetunes all models from the following article:

        Quistorp, P., Winn, T., Wolfrath, L., Dinsing, L., Taylor, T., & Gaikwad, R. (2025, August 13). 
        Detecting stereotypes (Unpublished manuscript). 
        University of Mannheim, Computational Analysis of Communication.
"""


from stereotype_detector.finetuning import BERT
from stereotype_detector.utils import load_data, get_huggingface_user_name
from stereotype_detector.config import DATA_PATH, MODEL_DIR_PATH

data = load_data(DATA_PATH)
user_name = get_huggingface_user_name()


# --- Model 1 -----
# Model: GBERT
# Finetuning: Fixed
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: False

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    use_default_hyperparameters=True,
    hub_model_id=f"{user_name}/gbert-fixed-heur-f-foca-t-free-f",
    output_dir=f"{MODEL_DIR_PATH}/gbert-fixed-heur-f-foca-f-free-f",
)

# ---- Model 2 ---
# Model: GBERT
# Finetuning: Optimiztion with Optuna
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: False

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id=f"{user_name}/gbert-cv-heur-f-foca-f-free-f",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/gbert-cv-heur-f-foca-f-free-f",
    use_focal_loss=False,
    heursitic_filtering=False,
    enable_layer_freezing=False,
)

# --- Model 3 ---
# Model: GBERT
# Finetuning: Optimiztion with Optuna
# Heuristic Filtering: True
# Focal loss: False
# Layer Freezing: False

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id=f"{user_name}/gbert-cv-heur-t-foca-f-free-f",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/gbert-cv-heur-t-foca-f-free-f",
    use_focal_loss=False,
    heursitic_filtering=True,
    enable_layer_freezing=False,
)

# --- Model 4 ----
# Model: GBERT
# Finetuning: Optimiztion with Optuna
# Heuristic Filtering: True
# Focal loss: True
# Layer Freezing: False

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id=f"{user_name}/gbert-cv-heur-t-foca-t-free-f",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/gbert-cv-heur-t-foca-t-free-f",
    use_focal_loss=True,
    heursitic_filtering=True,
    enable_layer_freezing=False,
)

# --- Model 5 ----
# Model: GBERT
# Finetuning: Optimiztion with Optuna
# Heuristic Filtering: False
# Focal loss: True
# Layer Freezing: False

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id=f"{user_name}/gbert-cv-heur-f-foca-t-free-f",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/gbert-cv-heur-f-foca-t-free-f",
    use_focal_loss=True,
    heursitic_filtering=False,
    enable_layer_freezing=False,
)

# --- Model 6 ----
# Model: GBERT
# Finetuning: Optimiztion with Optuna
# Heuristic Filtering: False
# Focal loss: True
# Layer Freezing: True

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id=f"{user_name}/gbert-cv-heur-f-foca-t-free-t",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/gbert-cv-heur-f-foca-t-free-t",
    use_focal_loss=True,
    heursitic_filtering=False,
    enable_layer_freezing=True,
)

# --- Model 7 ----
# Model: GBERT
# Finetuning: Optimiztion with Optuna
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: True

BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id=f"{user_name}/gbert-cv-heur-f-foca-f-free-t",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/gbert-cv-heur-f-foca-f-free-t",
    use_focal_loss=False,
    heursitic_filtering=False,
    enable_layer_freezing=True,
)


# --- Model 8 ---
# Model: BERT base multilingual
# Finetuning: Fixed
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: False

BERT.finetune(
    model="bert-base-multilingual-cased",
    data=data,
    use_default_hyperparameters=True,
    hub_model_id=f"{user_name}/gbert-fixed-heur-f-foca-t-free-f",
    output_dir=f"{MODEL_DIR_PATH}/gbert-fixed-heur-f-foca-f-free-f",
)

# --- Model 9 ---
# Model: BERT base multilingual
# Finetuning: Optimization with Optuna
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: False

BERT.finetune(
    model="bert-base-multilingual-cased",
    data=data,
    hub_model_id=f"{user_name}/multi-cv-heur-f-foca-f-free-f",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/multi-cv-heur-f-foca-f-free-f",
    use_focal_loss=False,
    heursitic_filtering=False,
    enable_layer_freezing=False,
)

# ---- Model 10 ----
# Model: BERT base multilingual
# Finetuning: Optimization with Optuna
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: True

BERT.finetune(
    model="bert-base-multilingual-cased",
    data=data,
    hub_model_id=f"{user_name}/multi-cv-heur-f-foca-f-free-t",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/multi-cv-heur-f-foca-f-free-t",
    use_focal_loss=False,
    heursitic_filtering=False,
    enable_layer_freezing=True,
)

# --- Model 11 ---
# Model: BERT base multilingual
# Finetuning: Optimization with Optuna
# Heuristic Filtering: False
# Focal loss: True
# Layer Freezing: True

BERT.finetune(
    model="bert-base-multilingual-cased",
    data=data,
    hub_model_id=f"{user_name}/multi-cv-heur-f-foca-t-free-t",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/multi-cv-heur-f-foca-t-free-t",
    use_focal_loss=True,
    heursitic_filtering=False,
    enable_layer_freezing=True,
)

# --- Model 12 ---
# Model: BERT base multilingual
# Finetuning: Optimization with Optuna
# Heuristic Filtering: False
# Focal loss: True
# Layer Freezing: False

BERT.finetune(
    model="bert-base-multilingual-cased",
    data=data,
    hub_model_id=f"{user_name}/multi-cv-heur-f-foca-t-free-f",
    evaluation_mode="cv",
    output_dir=f"{MODEL_DIR_PATH}/multi-cv-heur-f-foca-t-free-f",
    use_focal_loss=True,
    heursitic_filtering=False,
    enable_layer_freezing=False,
)
