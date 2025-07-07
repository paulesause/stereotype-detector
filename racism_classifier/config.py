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