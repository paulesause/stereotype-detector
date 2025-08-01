from racism_classifier.finetuning import BERT
from racism_classifier.utils import load_data

data = load_data("data/sample_paragraphs_1200.xlsx")

# holdout
print("""
# holdout
      """)
BERT.finetune(
        model="distilbert-base-uncased",
        data=data,
        hub_model_id="lwolfrat/test-focal-f-heuristic-t",
        evaluation_mode="holdout",
        output_dir="models/test-focal-f-heuristic-t",
        n_example_sample=10,
        use_focal_loss=False,
        heursitic_filtering=True,
)

