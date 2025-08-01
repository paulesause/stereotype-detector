from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict, ClassLabel
from racism_classifier.utils import load_data, get_huggingface_token,CustomTrainingArguments
from huggingface_hub import login, ModelCard, ModelCardData, HfApi
from racism_classifier.hyperparameter_optimization import make_model_init, compute_objective_BERT, optuna_hp_space_BERT, make_objective_BERT_cross_validation
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize, heuristic_filter_hf
from racism_classifier.evaluation import compute_evaluation_metrics
from racism_classifier.logger.metrics_logger import JsonlMetricsLoggerCallback
from sklearn.model_selection import KFold
from racism_classifier.config import  LABEL_COLUMN_NAME, NUMBER_OF_TRIALS, RANDOM_STATE, TEST_SPLIT_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, ALPHA, GAMMA, ID2LABEL_MAP, NUMBER_OF_LABELS
from racism_classifier.utils import FocalLossTrainer
import datetime
import optuna
import json
from pathlib import Path

def finetune(
        model:str,
        data: Dataset,
        output_dir:str,
        hub_model_id: str,
        evaluation_mode: str = "holdout",
        n_example_sample:int = None,
        heursitic_filtering: bool = False,
        use_focal_loss: bool = False,
        enable_layer_freezing: bool = True,
):
    # parameter check
    assert isinstance(hub_model_id, str), "paramter hub_model_id must be specified and a str."

    # Set Trainer Class to either default of custom
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
    
    data = data.map(rescale_warm_hot_dimension)#, batch=True)
    
    # Handling imbalanced Data
    if heursitic_filtering:
        data=heuristic_filter_hf(data)

    def _cast_labels_to_classlabel(ds):
        names = [ID2LABEL_MAP[i] for i in range(NUMBER_OF_LABELS)]
        if isinstance(ds, DatasetDict):
            for split in ds.keys():
                ds[split] = ds[split].cast_column(
                    LABEL_COLUMN_NAME, ClassLabel(num_classes=NUMBER_OF_LABELS, names=names)
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
    columns_to_remove = [col for col in data["train"].column_names if col not in columns_to_keep]

    data = data.map(tokenize(tokenizer),
                    batched=True,
                    batch_size=BATCH_SIZE,
                    remove_columns=columns_to_remove
                    )

    # -------------------------------------------------------------------------------------------
    # Fine-Tuning
    # -------------------------------------------------------------------------------------------

    # Hyperparameter Tuning
    if evaluation_mode == "holdout":
        # train-evaluation split
        train_validation = data["train"].train_test_split(test_size=TEST_SPLIT_SIZE)
        train_validation["validation"] = train_validation.pop("test")

        # training_args = CustomTrainingArguments(
        #     output_dir=output_dir,

        #     per_device_train_batch_size=BATCH_SIZE,
        #     per_device_eval_batch_size=BATCH_SIZE,
        #     num_train_epochs=EPOCHS,
        #     logging_strategy="epoch",
        #     hub_model_id=hub_model_id,
        #     logging_dir="logs",
        #     load_best_model_at_end=True,
        #     save_strategy="no",
        #     learning_rate=LEARNING_RATE,
        #     alpha=ALPHA,
        #     gamma=GAMMA,
        # )

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
            **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {})
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
            hp_space=lambda trial: optuna_hp_space_BERT(trial, use_focal_loss=use_focal_loss),
            compute_objective=compute_objective_BERT,
            n_trials=NUMBER_OF_TRIALS
            )
        # -------------------------------------------------------------------------------------
        # Model Testing
        # -------------------------------------------------------------------------------------

        # Set training args to best found during hyperparameter search
        for n, v in best_run.hyperparameters.items():
            if hasattr(training_args, n):
                setattr(training_args, n, v)
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        # Set training agrs to savte the model to the hub
        setattr(training_args, "save_strategy", "epoch")
        setattr(training_args, "eval_strategy", "epoch")
        setattr(training_args, "save_total_limit", 1)
        setattr(training_args, "save_strategy", "epoch")
        setattr(training_args, "hub_strategy", "end")
        setattr(training_args, "hub_private_repo", True)
        setattr(training_args, "push_to_hub", True)

        # freeze_embeddings = best_run.hyperparameters.get("freeze_embeddings", False)
        # num_transformer_layers_freeze = best_run.hyperparameters.get("num_transformer_layers_freeze", 0)
        if enable_layer_freezing:
            freeze_embeddings = best_run.hyperparameters.get("freeze_embeddings", False)
            num_transformer_layers_freeze = best_run.hyperparameters.get("num_transformer_layers_freeze", 0)
        else:
            freeze_embeddings = False
            num_transformer_layers_freeze = 0

        best_trainer = trainer_class(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(model, freeze_embeddings, num_transformer_layers_freeze),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()],
            )

        # Print gamma and alpha if using FocalLossTrainer
        # if isinstance(trainer, FocalLossTrainer):
        #     print(f"FocalLossTrainer gamma: {trainer.focal_loss.gamma}")
        #     print(f"FocalLossTrainer alpha: {trainer.focal_loss.alpha}")

    elif evaluation_mode == "cv":
        # hyperparameter tuning 
        study = optuna.create_study(direction="maximize", study_name="BERT_cross_validation")
        study.optimize(make_objective_BERT_cross_validation(model, data["train"], tokenizer, data_collator, trainer_class, use_focal_loss, enable_layer_freezing=enable_layer_freezing), n_trials=NUMBER_OF_TRIALS)

        best_params = study.best_params
        study_name = study.study_name

        # freeze_embeddings = best_params.get("freeze_embeddings", False)
        # num_transformer_layers_freeze = best_params.get("num_transformer_layers_freeze", 0)
        if enable_layer_freezing:
            freeze_embeddings = best_params.get("freeze_embeddings", False)
            num_transformer_layers_freeze = best_params.get("num_transformer_layers_freeze", 0)
        else:
            freeze_embeddings = False
            num_transformer_layers_freeze = 0

        # -------------------------------------------------------------------------------------
        # Model Testing
        # -------------------------------------------------------------------------------------

        # Train model with best params
        # training_args = CustomTrainingArguments(
        #     output_dir=output_dir,

        #     per_device_train_batch_size=4,
        #     per_device_eval_batch_size=4,
        #     num_train_epochs=1,
            
        #     save_strategy="epoch",
        #     eval_strategy="epoch",
        #     logging_strategy="epoch",
            
        #     hub_model_id=hub_model_id,
        #     logging_dir="logs",
        #     hub_strategy="end",
        #     hub_private_repo=True,
        #     push_to_hub=True,

        #     load_best_model_at_end=True,
        #     save_total_limit=1
        # )

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

             **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {})
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
            model_init=make_model_init(model, freeze_embeddings, num_transformer_layers_freeze),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()]
        )

    elif evaluation_mode == "nested_cv":

        outer_k = 5
        kf_outer = KFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)
        all_outer_scores = []
        for outer_train_index, outer_test_index in kf_outer.split(data["train"]):
            outer_train_data = data["train"].select(outer_train_index)
            outer_test_data = data["train"].select(outer_test_index)

            # Inner CV for hyperparameter tuning
            study = optuna.create_study(direction="maximize", study_name="BERT_nested_cross_validation")
            study.optimize(make_objective_BERT_cross_validation(model, outer_train_data, tokenizer, data_collator, trainer_class, use_focal_loss, enable_layer_freezing=enable_layer_freezing), n_trials=NUMBER_OF_TRIALS)

            best_params = study.best_params

            if enable_layer_freezing:
                freeze_embeddings = best_params.get("freeze_embeddings", False)
                num_transformer_layers_freeze = best_params.get("num_transformer_layers_freeze", 0)
            else:
                freeze_embeddings = False
                num_transformer_layers_freeze = 0

        #     # Train model with best params
        #     training_args = CustomTrainingArguments(
        #     output_dir=output_dir,

        #     per_device_train_batch_size=4,
        #     per_device_eval_batch_size=4,
        #     num_train_epochs=1,
            
        #     save_strategy="epoch",
        #     eval_strategy="epoch",
        #     logging_strategy="epoch",
            
        #     hub_model_id=hub_model_id,
        #     logging_dir="logs",
        #     hub_strategy="end",
        #     hub_private_repo=True,
        #     push_to_hub=True,

        #     load_best_model_at_end=True,
        #     save_total_limit=1
        # )


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

                 **({"alpha": ALPHA, "gamma": GAMMA} if use_focal_loss else {})
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
                model_init=make_model_init(model, freeze_embeddings, num_transformer_layers_freeze),
                data_collator=data_collator,
                compute_metrics=compute_evaluation_metrics,
                callbacks=[JsonlMetricsLoggerCallback()]
            )
    else:
        raise ValueError("Unsupported evaluation_mode parameter. Chose either 'holdout' or 'cv'.")
    
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

    # # Build a dictionary with requires_grad info for each parameter
    # freeze_status = {
    #     name: param.requires_grad
    #     for name, param in best_trainer.model.named_parameters()
    # }

    # # Also include the freeze settings if available
    # freeze_config = {
    #     "freeze_embeddings": freeze_embeddings,
    #     "num_transformer_layers_freeze": num_transformer_layers_freeze,
    #     "parameter_trainability": freeze_status
    # }

    # # Save to file
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    # with open(f"{output_dir}/freeze_config.json", "w") as f:
    #     json.dump(freeze_config, f, indent=4)

    best_trainer.push_to_hub(commit_message=commit_message)

#     from dataclasses import asdict
#     import json

#     with open("training_args.json", "w") as f:
#         json.dump(asdict(training_args), f, indent=4)

#     # Print the contents of training_args.json to the console
#     with open("training_args.json", "r") as f:
#         print("\n--- training_args.json contents ---")
#         print(f.read())

#     card_data = ModelCardData(
#     language='en',
#     license='mit',
#     library_name='transformers',
#     tags=['text-classification', 'layer-freezing', 'bert'],
#     base_model=model,
#     datasets='custom',
#     )

#     freeze_description = f"""
# ## Layer Freezing Details

# - `freeze_embeddings`: {freeze_embeddings}
# - `num_transformer_layers_freeze`: {num_transformer_layers_freeze}
# """

#     card = ModelCard.from_template(
#     card_data,
#     model_id=hub_model_id,
#     model_description=f"This BERT-based classifier was fine-tuned with the following layer freezing configuration:\n{freeze_description}",
#     )

#     readme_path = Path(output_dir) / "README.md"
#     card.save(readme_path)

#     # Upload the card manually via API
#     api = HfApi()
#     api.upload_file(
#     path_or_fileobj=str(readme_path),
#     path_in_repo="README.md",
#     repo_id=hub_model_id,
#     repo_type="model",
#     token=hugging_face_token,
#     commit_message="Updating model card with freezing details"
#     )

    readme_path = Path(output_dir) / "README.md"

    # 1) Load the auto-generated README (keeps metrics, tables, etc.)
    with open(readme_path, "r", encoding="utf-8") as f:
        card_text = f.read()

    # 2) Build sections to append
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

    # Scrub any secrets
    ta = {k: v for k, v in ta.items() if "token" not in k.lower()}

    training_args_section = "## ⚙️ TrainingArguments\n\n```json\n" + json.dumps(ta, indent=2, default=str) + "\n```"

    evaluation_section = "## 📊 Evaluation (from script)\n\n```json\n" + json.dumps(best_results, indent=2, default=str) + "\n```"

    # 3) Merge your sections **after** the autogenerated content
    merged = card_text.rstrip() + "\n\n" + freeze_section + "\n\n" + training_args_section + "\n\n" + evaluation_section + "\n"

    # 4) Save and upload only README.md to overwrite the one on the Hub
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(merged)

    HfApi().upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=hub_model_id,
        repo_type="model",
        token=hugging_face_token,
        commit_message="Append layer-freezing + TrainingArguments + eval to model card"
    )