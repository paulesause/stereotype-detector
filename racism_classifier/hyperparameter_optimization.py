from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoConfig

import numpy as np
from sklearn.model_selection import StratifiedKFold
from racism_classifier.config import (BERT_MODEL_NAME, 
                                      NUMBER_OF_LABELS, 
                                      ID2LABEL_MAP, 
                                      LABEL2ID_MAP, 
                                      RANDOM_STATE,
                                      NUMBER_CROSS_VALIDATION_FOLDS,
                                      DROP_OUT_RATE,
                                      HYPERPARAMETER_SPACE)
from racism_classifier.evaluation import compute_evaluation_metrics

def make_model_init(model_name: str=BERT_MODEL_NAME):
    def model_init():
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=NUMBER_OF_LABELS,
            id2label=ID2LABEL_MAP,
            label2id=LABEL2ID_MAP,
            hidden_dropout_prob=DROP_OUT_RATE,
            attention_probs_dropout_prob=DROP_OUT_RATE
        )
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
    return model_init

def optuna_hp_space_BERT(trial):
    hp = {}
    for name, params in HYPERPARAMETER_SPACE.items():
        ptype = params.get("type")
        if ptype == "float":
            hp[name] = trial.suggest_float(
                name,
                params["low"],
                params["high"],
                log=params.get("log", False)
            )
        elif ptype == "int":
            hp[name] = trial.suggest_int(
                name,
                params["low"],
                params["high"],
                log=params.get("log", False)
            )
        elif ptype == "categorical":
            hp[name] = trial.suggest_categorical(
                name,
                params["choices"]
            )
        else:
            raise ValueError(f"Unsupported hyperparameter type: {ptype}")
    return hp


def compute_objective_BERT(metrics):
    return metrics["eval_f1_macro"]


def make_objective_BERT_cross_validation(model, tokenized_dataset, tokenizer, data_collator):
    def objective_BERT_cross_validation(trial):
        """
        Applies Cross Validation together with the optuna library
        """
        # Trial parameters
        hp_space = optuna_hp_space_BERT(trial)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=NUMBER_CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        f1s = []

        y = tokenized_dataset["labels"]
        X = tokenized_dataset.remove_columns("labels")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_dataset = tokenized_dataset.select(train_idx)
            val_dataset = tokenized_dataset.select(val_idx)

            training_args = TrainingArguments(
                output_dir=f"./.tmp_results/fold{fold}",
                overwrite_output_dir=True,
                eval_strategy="epoch",
                save_strategy="no",
                per_device_eval_batch_size=4,
                disable_tqdm=True,
                load_best_model_at_end=False
            )

            for n, v in hp_space.items():
                setattr(training_args, n, v)

            trainer = Trainer(
                model=None,
                model_init=make_model_init(model),
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_evaluation_metrics
            )

            trainer.train()
            result = trainer.evaluate()
            f1s.append(result["eval_f1_macro"])

        return np.mean(f1s)

    return objective_BERT_cross_validation

