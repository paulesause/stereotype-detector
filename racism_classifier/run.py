from racism_classifier.finetuning import BERT
from racism_classifier.utils import load_data

data = load_data("data/holdout.xlsx")

# holdout
print(
    """
# holdout
      """
)
BERT.finetune(
    model="distilbert-base-uncased",
    data=data,
    hub_model_id="lwolfrat/test-nofreeze-modcard",
    evaluation_mode="holdout",
    output_dir="models/test-nofreeze-modcard",
    n_example_sample=10,
    enable_layer_freezing=False,
)
