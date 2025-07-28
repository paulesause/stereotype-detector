import numpy as np
import pandas as pd
from racism_classifier.racism_classifier.config import TEXT_COLUMN_NAME, COLD_COLUMN_NAME, WARM_COLUMN_NAME, LABEL_COLUMN_NAME


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

def tokenize(tokenizer):
    def f(batch):
        return tokenizer(batch[TEXT_COLUMN_NAME], truncation=True)
    return f


def lexical_diversity(text):
    """ Lexical diversity is the ratio of unique words to total words in a text.
    """
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0

def heuristic_filter(data):
    """Filter function to keep the 60% most diverse samples in the 'either' category.
      This downsamples without complete arbitrage"""
    # Apply only to "either" rows
    either_df = data[data['label'] == 'either'].copy()
    either_df['lex_div'] = either_df['text'].apply(lexical_diversity)

    # Keep top 60% based on lexical diversity
    filtered_either_df = either_df.nlargest(int(0.6 * len(either_df)), 'lex_div')

    # Combine and shuffle
    non_either_df = data[data['label'] != 'either']
    data_filtered = pd.concat([non_either_df, filtered_either_df])
    return data_filtered.sample(frac=1, random_state=42).reset_index(drop=True)