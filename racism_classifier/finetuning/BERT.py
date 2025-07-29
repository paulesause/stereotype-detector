from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from huggingface_hub import login
from racism_classifier.utils import load_data, get_huggingface_token
from racism_classifier.hyperparameter_optimization import make_model_init, compute_objective_BERT, optuna_hp_space_BERT, make_objective_BERT_cross_validation
from racism_classifier.preprocessing import rescale_warm_hot_dimension, tokenize, heuristic_filter_hf
from racism_classifier.evaluation import compute_evaluation_metrics
from racism_classifier.logger.metrics_logger import JsonlMetricsLoggerCallback
from racism_classifier.config import  LABEL_COLUMN_NAME, NUMBER_OF_TRIALS, RANDOM_STATE, TEST_SPLIT_SIZE, BATCH_SIZE
from sklearn.model_selection import KFold
import datetime
import optuna

def finetune(
        model:str,
        data: Dataset,
        output_dir:str,
        hub_model_id: str,
        evaluation_mode: str = "holdout",
        n_example_sample:int = None,
        heursitic_filtering: bool = False
):
    # parameter check
    assert isinstance(hub_model_id, str), "paramter hub_model_id must be specified and a str."

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
    # Train test split
    data = data.train_test_split(test_size=TEST_SPLIT_SIZE)
        
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

        training_args = TrainingArguments(
            output_dir=output_dir,

            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=1,
            logging_strategy="epoch",
            hub_model_id=hub_model_id,
            logging_dir="logs",
            load_best_model_at_end=True,
            save_strategy="no"
        )

        trainer = Trainer(
            model=None,
            args=training_args,
            train_dataset=train_validation["train"],
            eval_dataset=train_validation["validation"],
            tokenizer=tokenizer,
            model_init=make_model_init(model),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()]
        )

        print("--- Perform hyperparameter Search ---\n")

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

        # Set training args to best found during hyperparameter search
        for n, v in best_run.hyperparameters.items():
            setattr(training_args, n, v)

        # Set training agrs to savte the model to the hub
        setattr(training_args, "save_strategy", "epoch")
        setattr(training_args, "eval_strategy", "epoch")
        setattr(training_args, "save_total_limit", 1)
        setattr(training_args, "save_strategy", "epoch")
        setattr(training_args, "hub_strategy", "end")
        setattr(training_args, "hub_private_repo", True)
        setattr(training_args, "push_to_hub", True)

        best_trainer = Trainer(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(model),
            data_collator=data_collator,
            compute_metrics=compute_evaluation_metrics,
            callbacks=[JsonlMetricsLoggerCallback()]
        )

    elif evaluation_mode == "cv":
        # hyperparameter tuning 
        study = optuna.create_study(direction="maximize", study_name="BERT_cross_validation")
        study.optimize(make_objective_BERT_cross_validation(model, data["train"], tokenizer, data_collator), n_trials=NUMBER_OF_TRIALS)

        best_params = study.best_params
        study_name = study.study_name

        # -------------------------------------------------------------------------------------
        # Model Testing
        # -------------------------------------------------------------------------------------

        # Train model with best params
        training_args = TrainingArguments(
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
            save_total_limit=1
        )


        # Set training args to best found during hyperparameter search
        for n, v in best_params.items():
            setattr(training_args, n, v)


        best_trainer = Trainer(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(model),
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
            study.optimize(make_objective_BERT_cross_validation(model, outer_train_data, tokenizer, data_collator), n_trials=NUMBER_OF_TRIALS)

            best_params = study.best_params

            # Train model with best params
            training_args = TrainingArguments(
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
            save_total_limit=1
        )

            # Set training args to best found during hyperparameter search
            for n, v in best_params.items():
                setattr(training_args, n, v)

            best_trainer = Trainer(
            model=None,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            tokenizer=tokenizer,
            model_init=make_model_init(model),
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

    best_trainer.push_to_hub(commit_message=commit_message)
