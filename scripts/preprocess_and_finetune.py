from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from huggingface_hub import login
import evaluate
from racism_classifier.utils import load_data, get_huggingface_repro, get_huggingface_token
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize
from racism_classifier.evaluation import compute_evaluation_metrics
from racism_classifier.config import BERT_MODEL_NAME, MODEL_DIR_PATH, LABEL_COLUMN_NAME, DATA_PATH, ID2LABEL_MAP, LABEL2ID_MAP
import numpy as np


# ---------------------------------------------------------------------------------------------
# Logins
# ---------------------------------------------------------------------------------------------

hugging_face_token = get_huggingface_token()
login(token=hugging_face_token)

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
    
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=True)

columns_to_keep = [LABEL_COLUMN_NAME]
columns_to_remove = [col for col in train_test_icr_data["train"].column_names if col not in columns_to_keep]

tokenized_icr_data = train_test_icr_data.map(tokenize(tokenizer), 
                                             batched=True, 
                                             batch_size=8,
                                             remove_columns=columns_to_remove)

# small subset for testing
small_train = tokenized_icr_data["train"].shuffle(seed=42).select(range(20))
small_test = tokenized_icr_data["test"].shuffle(seed=42).select(range(20))


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4, id2label=ID2LABEL_MAP, label2id=LABEL2ID_MAP
)


training_args = TrainingArguments(
    output_dir=MODEL_DIR_PATH,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    push_to_hub=True,
    hub_model_id=get_huggingface_repro(),
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_icr_data["train"],
    eval_dataset=tokenized_icr_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_evaluation_metrics,
)

trainer.train()
trainer.evaluate()
trainer.push_to_hub()

