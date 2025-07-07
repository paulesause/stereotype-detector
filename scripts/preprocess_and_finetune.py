from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import evaluate
from racism_classifier.utils import load_data
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize
from racism_classifier.config import BERT_MODEL_NAME, MODEL_DIR_PATH, LABEL_COLUMN_NAME, DATA_PATH
import numpy as np

# ----------------------------------------------------------------------------------------------
# Data loding
# ---------------------------------------------------------------------------------------------

icr_data = load_data(DATA_PATH)

# -----------------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------------

# Rescaling hot warm dimension
rescaled_icr_data = icr_data.map(rescale_warm_hot_dimension, batched=True)

# Train test split
train_test_icr_data = rescaled_icr_data.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

columns_to_keep = [LABEL_COLUMN_NAME]
columns_to_remove = [col for col in train_test_icr_data["train"].column_names if col not in columns_to_keep]

tokenized_icr_data = train_test_icr_data.map(tokenize, 
                                             batched=True, 
                                             batch_size=8,
                                             remove_columns=columns_to_remove)

# small subset for testing
small_train = tokenized_icr_data["train"].shuffle(seed=42).select(range(40))
small_test = tokenized_icr_data["test"].shuffle(seed=42).select(range(40))


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "cold = 0 AND warm = 0", 1: "cold = 1 AND warm = 0", 2: "cold = 0 AND warm = 1", 3: "cold = 1 AND warm = 1"}
label2id = {"cold = 0 AND warm = 0": 0, "cold = 1 AND warm = 0": 1, "cold = 0 AND warm = 1": 2, "cold = 1 AND warm = 1": 3}

model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=MODEL_DIR_PATH,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    push_to_hub=False,
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_icr_data["train"],
    eval_dataset=tokenized_icr_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
