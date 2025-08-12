from racism_classifier.finetuning import BERT
from racism_classifier.utils import load_data

data = load_data("data/sample_paragraphs_1200.xlsx")

# holdout
print(
    """
# holdout
      """
)
BERT.finetune(
    model="deepset/gbert-base",
    data=data,
    hub_model_id="lwolfrat/test-deepset-gbert-base",
    evaluation_mode="holdout",
    output_dir="models/test-deepset-gbert-base",
    n_example_sample=10,
    use_focal_loss=False,
    heursitic_filtering=True,
    enable_layer_freezing=True,
)
