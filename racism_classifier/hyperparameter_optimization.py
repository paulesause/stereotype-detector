from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from racism_classifier.config import (
    NUMBER_OF_LABELS,
    ID2LABEL_MAP,
    LABEL2ID_MAP,
    RANDOM_STATE,
    NUMBER_CROSS_VALIDATION_FOLDS,
    DROP_OUT_RATE,
    HYPERPARAMETER_SPACE_BASE,
    HYPERPARAMETER_SPACE_FOCAL,
)
from racism_classifier.evaluation import compute_evaluation_metrics
from racism_classifier.utils import freeze_layers


def make_model_init(
    model_name: str = None,
    freeze_embeddings: bool = False,
    num_transformer_layers_freeze: int = 0,
):
    def model_init():
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=NUMBER_OF_LABELS,
            id2label=ID2LABEL_MAP,
            label2id=LABEL2ID_MAP,
            hidden_dropout_prob=DROP_OUT_RATE,
            attention_probs_dropout_prob=DROP_OUT_RATE,
            problem_type="single_label_classification",
        )
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )

        freeze_layers(loaded_model, freeze_embeddings, num_transformer_layers_freeze)

        return loaded_model

    return model_init


def _max_freezable_layers(model_name: str) -> int:
    if model_name == "distilbert-base-uncased":
        return 6
    if model_name in ("deepset/gbert-base", "bert-base-multilingual-cased"):
        return 12
    return 12


def optuna_hp_space_BERT(
    trial, model_name: str, use_focal_loss: bool, enable_layer_freezing=True
):
    space = HYPERPARAMETER_SPACE_FOCAL if use_focal_loss else HYPERPARAMETER_SPACE_BASE
    hp = {}
    for name, params in space.items():
        if not enable_layer_freezing and name in [
            "freeze_embeddings",
            "num_transformer_layers_freeze",
        ]:
            continue
        if name == "num_transformer_layers_freeze" and enable_layer_freezing:
            params = {**params, "high": _max_freezable_layers(model_name)}
        ptype = params.get("type")
        if ptype == "float":
            hp[name] = trial.suggest_float(
                name, params["low"], params["high"], log=params.get("log", False)
            )
        elif ptype == "int":
            hp[name] = trial.suggest_int(
                name, params["low"], params["high"], log=params.get("log", False)
            )
        elif ptype == "categorical":
            hp[name] = trial.suggest_categorical(name, params["choices"])
        elif ptype == "bool":
            hp[name] = trial.suggest_categorical(name, [True, False])
        else:
            raise ValueError(f"Unsupported hyperparameter type: {ptype}")
    return hp


def compute_objective_BERT(metrics):
    return metrics["eval_f1_macro"]


def make_objective_BERT_cross_validation(
    model,
    tokenized_dataset,
    tokenizer,
    data_collator,
    trainer_class,
    use_focal_loss: bool,
    enable_layer_freezing=True,
):
    def objective_BERT_cross_validation(trial):
        """
        Applies Cross Validation together with the optuna library
        """
        # Trial parameters
        hp_space = optuna_hp_space_BERT(
            trial,
            model_name=model,
            use_focal_loss=use_focal_loss,
            enable_layer_freezing=enable_layer_freezing,
        )

        # Get layer freezing parameters
        freeze_embeddings = hp_space.get("freeze_embeddings", False)
        num_transformer_layers_freeze = hp_space.get("num_transformer_layers_freeze", 0)

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=NUMBER_CROSS_VALIDATION_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
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
                load_best_model_at_end=False,
            )

            for n, v in hp_space.items():
                setattr(training_args, n, v)

            trainer = trainer_class(
                model=None,
                model_init=make_model_init(
                    model, freeze_embeddings, num_transformer_layers_freeze
                ),
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_evaluation_metrics,
            )

            trainer.train()
            result = trainer.evaluate()
            f1s.append(result["eval_f1_macro"])

        return np.mean(f1s)

    return objective_BERT_cross_validation
