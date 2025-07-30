from racism_classifier.finetuning import BERT
from racism_classifier.utils import load_data

data = load_data("data/holdout.xlsx")

transformer_layers = ['transformer.layer.0', 'transformer.layer.1', 'transformer.layer.2', 'transformer.layer.3', 'transformer.layer.4', 'transformer.layer.5']

# holdout
print("""
# small data set
# holdout
      """)

for i in range(len(transformer_layers)+1):
    for j in (True, False):
        frozen_layers = transformer_layers[:i]
        if j:
            frozen_layers = ["embedding"] + frozen_layers
        mod_id = "lwolfrat/distil-b-un_h_10_free_" + str(i) + ("_em" if j else "")
        output_dir = "models/distil-b-un_h_10_free_" + str(i) + ("_em" if j else "")
        BERT.finetune(
            model="distilbert-base-uncased",
            data=data,
            hub_model_id=mod_id,
            evaluation_mode="holdout",
            output_dir=output_dir,
            n_example_sample=10,
            frozen_layers=frozen_layers
        )

# cross-validation
print("""
# small data set
# cross-validation
      """)

for i in range(len(transformer_layers)+1):
    for j in (True, False):
        frozen_layers = transformer_layers[:i]
        if j:
            frozen_layers = ["embedding"] + frozen_layers
        mod_id = "lwolfrat/distil-b-un_h_10_free_" + str(i) + ("_em" if j else "")
        output_dir = "models/distil-b-un_h_10_free_" + str(i) + ("_em" if j else "")
        BERT.finetune(
            model="distilbert-base-uncased",
            data=data,
            hub_model_id=mod_id,
            evaluation_mode="cv",
            output_dir=output_dir,
            n_example_sample=10,
            frozen_layers=frozen_layers
        )
