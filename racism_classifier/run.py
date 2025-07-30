from racism_classifier.finetuning import BERT
from racism_classifier.utils import load_data

# This script entails layer freezing in a simple way.
# This means, we run the finetune() function multiple times, each time freezing different layers or none et all.

# insert own data path here
data = load_data("data/holdout.xlsx") 

# Those are the transformer layers of the DistilBERT model.
transformer_layers = ['transformer.layer.0', 'transformer.layer.1', 'transformer.layer.2', 'transformer.layer.3', 'transformer.layer.4', 'transformer.layer.5']

# holdout
print("""
# small data set
# holdout
      """)

# We loop over the transformer layers, with each iteration freezing one more layer (i.e. Iteration one: no transformer layers frozen, iteration two: first transformer layer frozen, iteration three: first 6 second transformer layer frozen, ...)
for i in range(len(transformer_layers)+1):
    # We also loop over the boolean value of whether to freeze the embedding layer or not. )
    for j in (True, False):
        # freeze the first i transformer layers
        frozen_layers = transformer_layers[:i] 
        # if j is True, we also freeze the embedding layer
        if j: 
            frozen_layers = ["embedding"] + frozen_layers
        # dynamic model id and output directory (change as needed)
        mod_id = "lwolfrat/distil-b-un_h_10_free_" + str(i) + ("_em" if j else "")
        output_dir = "models/distil-b-un_h_10_free_" + str(i) + ("_em" if j else "")
        # run the finetune function with the specified parameters
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
