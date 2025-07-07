import numpy as np
from transformers import AutoTokenizer
from racism_classifier.config import TEXT_COLUMN_NAME, BERT_MODEL_NAME, COLD_COLUMN_NAME, WARM_COLUMN_NAME, LABEL_COLUMN_NAME


def rescale_warm_hot_dimension(batch):
    """Rescale the Hot-Warm Dimensions to a singular scale.
    Follows the coding:

    0: cold = 0 AND warm = 0
    1: cold = 1 AND warm = 0
    2: cold = 0 AND warm = 1
    3: cold = 1 AND warm = 1    
    """

    cold = np.array(batch[COLD_COLUMN_NAME])
    warm = np.array(batch[WARM_COLUMN_NAME])
    rescaled = 2 * warm + cold

    return {LABEL_COLUMN_NAME: rescaled.tolist()}


tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch[TEXT_COLUMN_NAME], truncation=True)
