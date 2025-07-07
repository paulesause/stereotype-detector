# Racism classifier

This Project classifies paragraphs news articles according to the stereotype content model employing a BERT classifiert.

## Setup
### Clone Repository

Clone the repository by running

```bash
git clone https://github.com/paulesause/racism_classifier
```

### Install Dependencies
1. Create virtualenvironment

```bash
virtualenv venv
```

2.Activate virtualenvironment 

```bash
source venv/bin/activate
```

3. Install dependencies from requirements.txt file

```bash
pip install -r requirements.txt
```

4. Install local racism_classifier package in editable mode

```
pip install -e .
```
### Data

Create a `data` folder in the root of the repository. Place a `.xlsx` file with your labled data in this `data` folder.
The `.xlsx` sould contain the following columns:

- **`text_block`**: Containing the new acticle paragraphs
- **`warm`**: labled either 1 if the group of interst is described as warm or 0 if not
- **`cold`**: labled either 1 if the group of interst is described as cold or 0 if not 

# Usage
## Finetune BERT Model

Finetune the BERT Model and apply the previous preprocessing steps by running

```python
python3 scripts/preprocess_and_finetune.py
```