from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os
import json

class JsonlMetricsLoggerCallback(TrainerCallback):
    def __init__(self, filename="all_metrics.jsonl"):
        self.filename = filename
        self.filepath = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics_dir = os.path.join(args.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        self.filepath = os.path.join(metrics_dir, self.filename)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        step = state.global_step
        metrics["step"] = step
        metrics["checkpoint_path"] = os.path.join(args.output_dir, f"checkpoint-{step}")

        with open(self.filepath, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        print(f"📋 Appended metrics for step {step} to {self.filepath}")
