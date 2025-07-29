import pandas as pd
data=pd.read_excel("C:/Users/Theo/Downloads/sample_paragraphs_1200.xlsx")
data = Dataset.from_pandas(data)
from racism_classifier.preprocessing import heuristic_filter
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize
from datasets import Dataset, concatenate_datasets

def lexical_diversity(text):
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0

def apply_lexical_diversity(example):
    return {"lex_div": lexical_diversity(example["text_block"])}

a = [{"lex_div": lexical_diversity(data["text_block"][i])} for i in range(len(data["text_block"]))]

def heuristic_filter_hf(dataset):
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
data_filtered = heuristic_filter_hf(data)