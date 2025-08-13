# Stereotype Detector

This Project classifies paragraphs from news articles as warm/cold/neutral according to the Stereotype Content Model (Fiske, 2018) by employing different BERT classifiert.

## Setup
### Clone Repository

Clone the repository by running

```bash
git clone https://github.com/paulesause/stereotype-detector
```

### Install Dependencies

1. Create virtualenvironment

```bash
virtualenv venv
```

2. Activate virtualenvironment 

```bash
source venv/bin/activate
```

3. Install dependencies from requirements.txt file

```bash
pip install -r requirements.txt
```

4. Install local stereotype_detector package in editable mode

```
pip install -e .
```
### Data

Create a `data` folder in the root of the repository. Place a `.xlsx` file with your labled data in this `data` folder.
Then change the `DATA_PATH` variable in `stereotype_detector/config.py` to the path of your `.xlsx` file.

The `.xlsx` sould contain the following columns:

- **`text_block`**: Containing the new acticle paragraphs
- **`warm`**: labled either 1 if the group of interst is described as warm or 0 if not
- **`cold`**: labled either 1 if the group of interst is described as cold or 0 if not 

All labeling shout be done with respect to the Stereotype Conent Model (Fiske, 2018).

### Environment Variables

Setup the following environment variables in  a `.env` file at the root of the repository.


**Hugging Face Access Token**

```
HUGGING_FACE_TOKEN=<your-huggingface-access-token>
```

**Hugging Face User Name**

```
HUGGING_FACE_USER_NAME=<your-huggingface-user-name>
```

## Usage
### Finetuning a BERT Classifier

The primary method for finetuning is `finetune(...)`, located in `stereotype_detector/finetuning/BERT.py`

This method handles end-to-end training of a BERT-based classifier using the Hugging Face Transformers ecosystem. It supports multiple evaluation strategies, integrates hyperparameter tuning via Optuna, and pushes the trained model to the Hugging Face Hub.

#### `finetune(...)` Overview

This method trains a transformer-based classification model with support for:
- Holdout, Cross-Validation, or Nested Cross-Validation
- Hyperparameter tuning (via Optuna)
- Focal loss, layer freezing, and heuristic filtering
- Custom training args or a default configuration according to Jumle et al. (2025)
- Model and metric logging to Hugging Face Hub

##### Parameters

| Name                          | Type                       | Description                                                                                         |
| ----------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------- |
| `model`                       | `str`                      | Name or path of the pre-trained Hugging Face model (e.g., `"bert-base-uncased"`). **Required.**     |
| `data`                        | `Dataset` or `DatasetDict` | Training dataset. Must include label columns.                                                       |
| `output_dir`                  | `str`                      | Directory to save logs, checkpoints, and outputs. **Required.**                                     |
| `hub_model_id`                | `str`                      | Repository name on the Hugging Face Hub. **Required.**                                              |
| `evaluation_mode`             | `str`                      | One of: `"holdout"` (default), `"cv"`, or `"nested_cv"`. Determines evaluation and tuning strategy. |
| `use_default_hyperparameters` | `bool`                     | Use preset hyperparameters from Jumle et al. (2025).                                                |
| `n_example_sample`            | `int`                      | Sample `n` examples for faster debugging. Optional.                                                 |
| `heuristic_filtering`         | `bool`                     | Apply heuristics to filter data before training.                                                    |
| `use_focal_loss`              | `bool`                     | Replace standard cross-entropy with focal loss.                                                     |
| `enable_layer_freezing`       | `bool`                     | Freeze embedding and transformer layers during training.                                            |


### Example Usage

```python
from stereotype_detector.finetuning import BERT
from stereotype_detector.utils import load_data, get_huggingface_user_name
from stereotype_detector.config import DATA_PATH, MODEL_DIR_PATH

data = load_data(DATA_PATH)
user_name = get_huggingface_user_name()


# --- Model 1 -----
# Model: GBERT
# Finetuning: Fixed
# Heuristic Filtering: False
# Focal loss: False
# Layer Freezing: False

BERT.finetune(
        model="deepset/gbert-base",
        data=data,
        use_default_hyperparameters=True,
        hub_model_id=f"{user_name}/gbert-fixed-heur-f-foca-f-free-f",
        output_dir=f"{MODEL_DIR_PATH}/gbert-fixed-heur-f-foca-f-free-f"
)
```

### Scripts

All models mentioned in Quistorp et al. (2025) can be trained by running:

```bash
python3 scripts/finetune.py
```
However, since this may take up exhaustive computational resources consider splitting it in different parts.

## References
   
    Fiske, S. T. (2018). Stereotype Content: 
    Warmth and Competence Endure. Current Directions in Psychological Science,  
    27(2), 67–73. https://doi.org/10.1177/0963721417738825

    Jumle, V., Makhortykh, M., Sydorova, M., & Vziatysheva, V. (2025).
    Finding Frames With BERT: A Transformer-Based Approach to Generic News Frame Detection.
    Social Science Computer Review. https://doi.org/10.1177/08944393251338396

    Quistorp, P., Winn, T., Wolfrath, L., Dinsing, L., Taylor, T., & Gaikwad, R. (2025, August 13). 
    Detecting stereotypes (Unpublished manuscript). 
    University of Mannheim, Computational Analysis of Communication.
