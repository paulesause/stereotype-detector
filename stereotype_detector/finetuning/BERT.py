from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict, ClassLabel
from stereotype_detector.utils import (
    get_huggingface_token,
    CustomTrainingArguments,
)
from huggingface_hub import login, HfApi
from stereotype_detector.hyperparameter_optimization import (
    make_model_init,
    compute_objective_BERT,
    optuna_hp_space_BERT,
    make_objective_BERT_cross_validation,
)
from stereotype_detector.preprocessing import (
    rescale_warm_hot_dimension,
    tokenize,
    heuristic_filter_hf,
)
from stereotype_detector.evaluation import compute_evaluation_metrics
from stereotype_detector.logger.metrics_logger import JsonlMetricsLoggerCallback
from sklearn.model_selection import KFold
from stereotype_detector.config import (
    LABEL_COLUMN_NAME,
    NUMBER_OF_TRIALS,
    RANDOM_STATE,
    TEST_SPLIT_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    ALPHA,
    GAMMA,
    ID2LABEL_MAP,
    NUMBER_OF_LABELS,
)
from stereotype_detector.utils import FocalLossTrainer
import datetime
import optuna
import json
from pathlib import Path


def finetune(
    model: str,
    data: Dataset,
    output_dir: str,
    hub_model_id: str,
    evaluation_mode: str = "holdout",
    use_default_hyperparameters=False,
    n_example_sample: int = None,
    heursitic_filtering: bool = False,
    use_focal_loss: bool = False,
    enable_layer_freezing: bool = True,
):
    """
    Fine-tunes a transformer model on a given dataset, with optional hyperparameter tuning.

    Args:
        model (str): Name or path of the Hugging Face model to fine-tune.
        data (Dataset): A Hugging Face `datasets.Dataset` or `DatasetDict` containing the training data.
        output_dir (str): Path to save the model and results.
        hub_model_id (str): Hugging Face Hub model repository ID.
        evaluation_mode (str, optional): One of {"holdout", "cv", "nested_cv"}.
            Determines the evaluation strategy. Defaults to "holdout".
        n_example_sample (int, optional): If provided, samples this many examples from the dataset.
        heuristic_filtering (bool, optional): Whether to apply heuristic filtering to the dataset.
        use_focal_loss (bool, optional): Whether to use focal loss instead of cross-entropy.
        enable_layer_freezing (bool, optional): Whether to freeze embedding and transformer layers during training.
        use_default_hyperparameters (bool, optional): If True, uses hyperparameters as suggested by Jumle et al. (2025).
            Although Jumle et al. (2025) employed dynamic warmup steps and cosine
            annealing, these features are not available in the Transformers library and ommitted with this parameter.

            Jumle, V., Makhortykh, M., Sydorova, M., & Vziatysheva, V. (2025).
            Finding Frames With BERT: A Transformer-Based Approach to Generic News Frame Detection.
            Social Science Computer Review. https://doi.org/10.1177/08944393251338396

    """

    # parameter check
    assert isinstance(
        hub_model_id, str
    ), "paramter hub_model_id must be specified and a str."

    # Set Trainer Class to either default or custom
    trainer_class = FocalLossTrainer if use_focal_loss else Trainer

    ArgsCls = CustomTrainingArguments if use_focal_loss else TrainingArguments
    # ---------------------------------------------------------------------------------------------
    # Logins
    # ---------------------------------------------------------------------------------------------

    # Huggingface login
    hugging_face_token = get_huggingface_token()
    login(token=hugging_face_token)

    # ----------------------------------------------------------------------------------------------
    # Data loding
    # ----------------------------------------------------------------------------------------------

    if n_example_sample:
        data = data.shuffle(seed=RANDOM_STATE).select(range(n_example_sample))

    # -----------------------------------------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------------------------------------

    # Rescaling hot warm dimension

    data = data.map(rescale_warm_hot_dimension)  # , batch=True)

    # Handling imbalanced Data
    if heursitic_filtering:
        data = heuristic_filter_hf(data)

    def _cast_labels_to_classlabel(ds):
        names = [ID2LABEL_MAP[i] for i in range(NUMBER_OF_LABELS)]
        if isinstance(ds, DatasetDict):
            for split in ds.keys():
                ds[split] = ds[split].cast_column(
                    LABEL_COLUMN_NAME,
                    ClassLabel(num_classes=NUMBER_OF_LABELS, names=names),
                )
            return ds
        else:
            return ds.cast_column(
                LABEL_COLUMN_NAME, ClassLabel(num_classes=NUMBER_OF_LABELS, names=names)
            )

    # Train test split
    data = data.train_test_split(test_size=TEST_SPLIT_SIZE)

    data = _cast_labels_to_classlabel(data)

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    columns_to_keep = [LABEL_COLUMN_NAME]
    columns_to_remove = [
        col for col in data["train"].column_names if col not in columns_to_keep
    ]

    data = data.map(
        tokenize(tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=columns_to_remove,
    )

    # -------------------------------------------------------------------------------------------
    # Fine-Tuning
    # -------------------------------------------------------------------------------------------

    if use_default_hyperparameters:
        print("--- using default hyperparameters --- \n")

        freeze_embeddings = False
        enable_layer_freezing = False
        num_transformer_layers_freeze = 0
        heursitic_filtering = False
        use_focal_loss = True

        trainer_class = FocalLossTrainer
        ArgsCls = CustomTrainingArguments

        training_args = ArgsCls(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            hub_model_id=hub_model_id,
            logging_dir="logs",
            hub_strategy="end",
            hub_private_repo=True,
            push_to_hub=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {}),
        )

        best_trainer = trainer_class(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(
                model, freeze_embeddings, num_transformer_layers_freeze
            ),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()],
        )

    # Hyperparameter Tuning
    elif evaluation_mode == "holdout":
        # train-evaluation split
        train_validation = data["train"].train_test_split(test_size=TEST_SPLIT_SIZE)
        train_validation["validation"] = train_validation.pop("test")

        training_args = ArgsCls(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            logging_strategy="epoch",
            hub_model_id=hub_model_id,
            logging_dir="logs",
            load_best_model_at_end=True,
            save_strategy="no",
            learning_rate=LEARNING_RATE,
            # only include these when focal is ON
            **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {}),
        )

        trainer = trainer_class(
            model=None,
            args=training_args,
            train_dataset=train_validation["train"],
            eval_dataset=train_validation["validation"],
            tokenizer=tokenizer,
            model_init=make_model_init(model),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()],
        )

        print("--- Perform hyperparameter Search ---\n")

        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=lambda trial: optuna_hp_space_BERT(
                trial,
                model_name=model,
                use_focal_loss=use_focal_loss,
                enable_layer_freezing=enable_layer_freezing,
            ),
            compute_objective=compute_objective_BERT,
            n_trials=NUMBER_OF_TRIALS,
        )
        # -------------------------------------------------------------------------------------
        # Model Testing
        # -------------------------------------------------------------------------------------

        # Set training args to best found during hyperparameter search
        for n, v in best_run.hyperparameters.items():
            if hasattr(training_args, n):
                setattr(training_args, n, v)

        # Set training agrs to savte the model to the hub
        setattr(training_args, "save_strategy", "epoch")
        setattr(training_args, "eval_strategy", "epoch")
        setattr(training_args, "save_total_limit", 1)
        setattr(training_args, "save_strategy", "epoch")
        setattr(training_args, "hub_strategy", "end")
        setattr(training_args, "hub_private_repo", True)
        setattr(training_args, "push_to_hub", True)

        # Set layer freezing parameters if enabled
        if enable_layer_freezing:
            freeze_embeddings = best_run.hyperparameters.get("freeze_embeddings", False)
            num_transformer_layers_freeze = best_run.hyperparameters.get(
                "num_transformer_layers_freeze", 0
            )
        else:
            freeze_embeddings = False
            num_transformer_layers_freeze = 0

        best_trainer = trainer_class(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(
                model, freeze_embeddings, num_transformer_layers_freeze
            ),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()],
        )

    elif evaluation_mode == "cv":
        # hyperparameter tuning
        study = optuna.create_study(
            direction="maximize", study_name="BERT_cross_validation"
        )
        study.optimize(
            make_objective_BERT_cross_validation(
                model,
                data["train"],
                tokenizer,
                data_collator,
                trainer_class,
                use_focal_loss,
                enable_layer_freezing=enable_layer_freezing,
            ),
            n_trials=NUMBER_OF_TRIALS,
        )

        best_params = study.best_params
        study_name = study.study_name

        # Set layer freezing parameters if enabled
        if enable_layer_freezing:
            freeze_embeddings = best_params.get("freeze_embeddings", False)
            num_transformer_layers_freeze = best_params.get(
                "num_transformer_layers_freeze", 0
            )
        else:
            freeze_embeddings = False
            num_transformer_layers_freeze = 0

        # -------------------------------------------------------------------------------------
        # Model Testing
        # -------------------------------------------------------------------------------------

        training_args = ArgsCls(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            hub_model_id=hub_model_id,
            logging_dir="logs",
            hub_strategy="end",
            hub_private_repo=True,
            push_to_hub=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {}),
        )

        # Set training args to best found during hyperparameter search
        for n, v in best_params.items():
            setattr(training_args, n, v)

        best_trainer = trainer_class(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(
                model, freeze_embeddings, num_transformer_layers_freeze
            ),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()],
        )

    elif evaluation_mode == "nested_cv":
        outer_k = 5
        kf_outer = KFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)
        all_outer_scores = []
        for outer_train_index, outer_test_index in kf_outer.split(data["train"]):
            outer_train_data = data["train"].select(outer_train_index)
            outer_test_data = data["train"].select(outer_test_index)

            # Inner CV for hyperparameter tuning
            study = optuna.create_study(
                direction="maximize", study_name="BERT_nested_cross_validation"
            )
            study.optimize(
                make_objective_BERT_cross_validation(
                    model,
                    outer_train_data,
                    tokenizer,
                    data_collator,
                    trainer_class,
                    use_focal_loss,
                    enable_layer_freezing=enable_layer_freezing,
                ),
                n_trials=NUMBER_OF_TRIALS,
            )

            best_params = study.best_params

            # set layer freezing parameters if enabled
            if enable_layer_freezing:
                freeze_embeddings = best_params.get("freeze_embeddings", False)
                num_transformer_layers_freeze = best_params.get(
                    "num_transformer_layers_freeze", 0
                )
            else:
                freeze_embeddings = False
                num_transformer_layers_freeze = 0

            training_args = ArgsCls(
                output_dir=output_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=1,
                save_strategy="epoch",
                eval_strategy="epoch",
                logging_strategy="epoch",
                hub_model_id=hub_model_id,
                logging_dir="logs",
                hub_strategy="end",
                hub_private_repo=True,
                push_to_hub=True,
                load_best_model_at_end=True,
                save_total_limit=1,
                **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {}),
            )

            # Set training args to best found during hyperparameter search
            for n, v in best_params.items():
                setattr(training_args, n, v)

            best_trainer = trainer_class(
                model=None,
                args=training_args,
                train_dataset=data["train"],
                eval_dataset=data["test"],
                tokenizer=tokenizer,
                model_init=make_model_init(
                    model, freeze_embeddings, num_transformer_layers_freeze
                ),
                data_collator=data_collator,
                compute_metrics=compute_evaluation_metrics,
                callbacks=[JsonlMetricsLoggerCallback()],
            )
    else:
        raise ValueError(
            "Unsupported evaluation_mode parameter. Chose either 'holdout' or 'cv'."
        )

    # Train with best hyperparmeters
    print("--- Train with best hyperparameters ---\n")
    best_trainer.train()

    # Evaluate
    print("--- Test best model ---\n")
    best_results = best_trainer.evaluate(eval_dataset=data["test"])

    # Push to hub
    print("--- push best model and tokenizer to hub ---\n")
    current_time = str(datetime.datetime.now()).replace(" ", "_")
    commit_message = f"End-training-{current_time}"

    best_trainer.push_to_hub(commit_message=commit_message)

    # ----- Append layer freezing information and training arguments to model card -----

    readme_path = Path(output_dir) / "README.md"

    # Load the auto-generated README
    with open(readme_path, "r", encoding="utf-8") as f:
        card_text = f.read()

    # Build sections to append
    freeze_section = f"""
## 🔒 Layer Freezing

- `freeze_embeddings`: {freeze_embeddings}
- `num_transformer_layers_freeze`: {num_transformer_layers_freeze}
"""

    # Prefer .to_dict() for TrainingArguments; fall back to dataclasses if needed
    try:
        ta = training_args.to_dict()
    except Exception:
        from dataclasses import asdict

        ta = asdict(training_args)

    # Remove token-related fields from TrainingArguments
    ta = {k: v for k, v in ta.items() if "token" not in k.lower()}

    training_args_section = (
        "## ⚙️ TrainingArguments\n\n```json\n"
        + json.dumps(ta, indent=2, default=str)
        + "\n```"
    )

    evaluation_section = (
        "## 📊 Evaluation (from script)\n\n```json\n"
        + json.dumps(best_results, indent=2, default=str)
        + "\n```"
    )

    # Merge the sections after the autogenerated content
    merged = (
        card_text.rstrip()
        + "\n\n"
        + freeze_section
        + "\n\n"
        + training_args_section
        + "\n\n"
        + evaluation_section
        + "\n"
    )

    # Save and upload only README.md to overwrite the one on the Hub
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(merged)

    HfApi().upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=hub_model_id,
        repo_type="model",
        token=hugging_face_token,
        commit_message="Append layer-freezing + TrainingArguments + eval to model card",
    )
