import numpy as np
import pandas as pd
from racism_classifier.config import TEXT_COLUMN_NAME, COLD_COLUMN_NAME, WARM_COLUMN_NAME, LABEL_COLUMN_NAME
from datasets import Dataset, concatenate_datasets

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
    rescaled = (2 * warm + cold).astype("int64")

    return {LABEL_COLUMN_NAME: rescaled.tolist()}

def tokenize(tokenizer):
    def f(batch):
        return tokenizer(batch[TEXT_COLUMN_NAME], truncation=True)
    return f
    
def lexical_diversity(text):
    """ Lexical diversity is the ratio of unique words to total words in a text.
    """
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0

def apply_lexical_diversity(example):
    return {"lex_div": lexical_diversity(example["text_block"])}

def heuristic_filter_hf(dataset):
    """Filter function to keep the 60% most diverse samples in the 'either' category.
      This downsamples without complete arbitrage"""
    # Step 1: Compute lexical diversity
    dataset = dataset.map(apply_lexical_diversity)

    # Step 2: Split into 'either' and 'not-either'
    either_ds = dataset.filter(lambda x: x["labels"] == 0)
    other_ds = dataset.filter(lambda x: x["labels"] != 0)

    # Step 3: Sort 'either' subset by lexical diversity
    either_ds = either_ds.sort("lex_div", reverse=True)

    # Step 4: Keep top 60% of 'either'
    top_n = int(0.6 * len(either_ds))
    filtered_either_ds = either_ds.select(range(top_n))

    # Step 5: Concatenate filtered + other
    final_dataset = concatenate_datasets([filtered_either_ds, other_ds]).shuffle(seed=42)

    return final_dataset