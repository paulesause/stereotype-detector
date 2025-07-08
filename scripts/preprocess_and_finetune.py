from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from huggingface_hub import login
from racism_classifier.utils import load_data, get_huggingface_repro, get_huggingface_token
from racism_classifier.hyperparameter_optimization import model_init_BERT, compute_objective_BERT, optuna_hp_space_BERT
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize
from racism_classifier.evaluation import compute_evaluation_metrics
from racism_classifier.logger.metrics_logger import JsonlMetricsLoggerCallback
from racism_classifier.config import BERT_MODEL_NAME, MODEL_DIR_PATH, LABEL_COLUMN_NAME, DATA_PATH, ID2LABEL_MAP, LABEL2ID_MAP, NUMBER_OF_TRIALS
import datetime
import os


# ---------------------------------------------------------------------------------------------
# Logins
# ---------------------------------------------------------------------------------------------

# Huggingface login
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

# -------------------------------------------------------------------------------------------
# Fine-Tuning
# -------------------------------------------------------------------------------------------

# small subset for testing
small_train = tokenized_icr_data["train"].shuffle(seed=42).select(range(10))
small_test = tokenized_icr_data["test"].shuffle(seed=42).select(range(10))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train-evaluation split
train_validation_small = small_train.train_test_split(test_size=0.2)
train_validation_small["validation"] = train_validation_small.pop("test")

model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4, id2label=ID2LABEL_MAP, label2id=LABEL2ID_MAP
)


training_args = TrainingArguments(
    output_dir=MODEL_DIR_PATH,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    
    hub_model_id=get_huggingface_repro(),
    logging_dir="logs",

    load_best_model_at_end=True,
    save_total_limit=1
)

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=train_validation_small["train"],
    eval_dataset=train_validation_small["validation"],
    tokenizer=tokenizer,
    model_init=model_init_BERT,
    data_collator=data_collator,
    compute_metrics=compute_evaluation_metrics,
    callbacks=[JsonlMetricsLoggerCallback()]
)

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space_BERT,
    compute_objective=compute_objective_BERT,
    n_trials=NUMBER_OF_TRIALS
    )

# -------------------------------------------------------------------------------------
# Model Testing
# -------------------------------------------------------------------------------------

# Load best run/model
best_run_path = f"./{MODEL_DIR_PATH}/run-{best_run.run_id}"
checkpoint = os.listdir(best_run_path)[0]
best_model_path = os.path.join(best_run_path, checkpoint)


best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
best_tokenizer = AutoTokenizer.from_pretrained(best_model_path)

best_trainer = Trainer(
    model=best_model,
    args=training_args,
    tokenizer=best_tokenizer,
    data_collator=data_collator,
    eval_dataset=small_test,
    compute_metrics=compute_evaluation_metrics
)

# Evaluate

best_results = best_trainer.evaluate(eval_dataset=small_test)

current_time = str(datetime.datetime.now()).replace(" ", "_")
commit_message = f"run-{best_run.run_id}-{current_time}"
best_trainer.push_to_hub(commit_message=commit_message)

from transformers import trainer_utils