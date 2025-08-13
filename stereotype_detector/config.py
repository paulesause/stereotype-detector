# Column Names
TEXT_COLUMN_NAME = "text_block"
COLD_COLUMN_NAME = "cold"
WARM_COLUMN_NAME = "warm"
ARTICLE_ID_COLUMN_NAME = "article_id"
PUB_COLUMN_NAME = "pub"
PAR_INDEX_NAME = "par_index"
LABEL_COLUMN_NAME = "labels"  # Do not change this name. Tranformers excpect them column with the labels named "labels"

# Paths
MODEL_DIR_PATH = "models"
DATA_PATH = "data/ICR_sample.xlsx"

# DATA Lables
ID2LABEL_MAP = {
    0: "cold = 0 AND warm = 0",
    1: "cold = 1 AND warm = 0",
    2: "cold = 0 AND warm = 1"
}
LABEL2ID_MAP = {
    "cold = 0 AND warm = 0": 0,
    "cold = 1 AND warm = 0": 1,
    "cold = 0 AND warm = 1": 2
}
NUMBER_OF_LABELS = 3

# Evaluation
TEST_SPLIT_SIZE = 0.2

# Finetuning
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5
ALPHA = None
GAMMA = 3.0

# Hyperparamter Search
NUMBER_OF_TRIALS = 2
RANDOM_STATE = 42
NUMBER_CROSS_VALIDATION_FOLDS = 3
DROP_OUT_RATE = 0.3

# Hyperparameter space
# Base HPO space (no focal-loss params)
HYPERPARAMETER_SPACE_BASE = {
    "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
    "per_device_train_batch_size": {"type": "categorical", "choices": [1, 2, 3, 4]},
    "num_train_epochs": {"type": "int", "low": 1, "high": 2},
    "warmup_steps": {"type": "int", "low": 1, "high": 100},
    "freeze_embeddings": {"type": "bool"},
    "num_transformer_layers_freeze": {"type": "int", "low": 0, "high": 6},
}

# Focal-loss extension
HYPERPARAMETER_SPACE_FOCAL = {
    **HYPERPARAMETER_SPACE_BASE,
    "gamma": {"type": "float", "low": 0.1, "high": 5.0, "log": True},
    "alpha": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
}
