import pandas as pd
data=pd.read_excel("C:/Users/Theo/Downloads/sample_paragraphs_1200.xlsx")
data = Dataset.from_pandas(data)
from racism_classifier.preprocessing import heuristic_filter
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize
from datasets import Dataset, concatenate_datasets

from racism_classifier.finetuning import BERT
from racism_classifier.utils import load_data

data = load_data("C:/Users/Theo/Downloads/sample_paragraphs_1200.xlsx")

# holdout
print("""
# holdout
      """)
BERT.finetune(
        model="distilbert-base-uncased",
        data=data,
        hub_model_id="lwolfrat/test-freeze",
        evaluation_mode="holdout",
        output_dir="models/test-freeze",
        n_example_sample=10,
        use_focal_loss= True #,
        #heursitic_filtering=True
)