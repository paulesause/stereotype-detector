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
        hub_model_id="lwolfrat/test-ncv-focal-t-heuristic-t-freeze-t",
        evaluation_mode="nested_cv",
        output_dir="models/test-ncv-focal-t-heuristic-t-freeze-t",
        n_example_sample=20,
        use_focal_loss=True,
        heursitic_filtering=True,
        enable_layer_freezing=True,
)

