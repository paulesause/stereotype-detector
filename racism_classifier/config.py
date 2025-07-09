# Model Names
BERT_MODEL_NAME = "distilbert-base-uncased"

# Column Names
TEXT_COLUMN_NAME = "text_block"
COLD_COLUMN_NAME = "cold"
WARM_COLUMN_NAME = "warm"
ARTICLE_ID_COLUMN_NAME = "article_id"
PUB_COLUMN_NAME = "pub"
PAR_INDEX_NAME = "par_index"
LABEL_COLUMN_NAME = "labels" # Do not change this name. Tranformers excpect them column with the labels named "labels"

# Paths
MODEL_DIR_PATH = "models"
DATA_PATH = "data/ICR_sample.xlsx"

# DATA Lables
ID2LABEL_MAP = {0: "cold = 0 AND warm = 0", 1: "cold = 1 AND warm = 0", 2: "cold = 0 AND warm = 1", 3: "cold = 1 AND warm = 1"}
LABEL2ID_MAP = {"cold = 0 AND warm = 0": 0, "cold = 1 AND warm = 0": 1, "cold = 0 AND warm = 1": 2, "cold = 1 AND warm = 1": 3}
NUMBER_OF_LABELS = 4

# Evaluation
TEST_SPLIT_SIZE = 0.2

# Finetuning
BATCH_SIZE = 4

# Hyperparamter Search
NUMBER_OF_TRIALS = 2
RANDOM_STATE=42
NUMBER_CROSS_VALIDATION_FOLDS=3

# Hyperparameter space
HYPERPARAMETER_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
    "per_device_train_batch_size": {"type": "categorical", "choices": [1, 2, 3, 4]},
    "num_train_epochs": {"type": "int", "low": 1, "high": 2}
}
