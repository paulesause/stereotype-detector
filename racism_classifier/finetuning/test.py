from racism_classifier.finetuning.BERT import finetune
import numpy as np
import pandas as pd
dataset=pd.read_csv("C:/Users/Theo/Downloads/ICR_sample.csv")
finetune("roberta-base",
        dataset,
        "C:/Users/Theo/Downloads",
        "Twinn/BERT-NewsClassifier",
        "nested_cv",
        10)