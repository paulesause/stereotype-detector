from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import krippendorff


def compute_evaluation_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", beta=1
    )
    krippendorff_alpha = krippendorff_alpha_score(labels, preds)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "krippendorffs_alpha": krippendorff_alpha,
    }


def krippendorff_alpha_score(labels, preds):
    reliability_data = np.vstack([labels, preds])
    try:
        return krippendorff.alpha(reliability_data, level_of_measurement="nominal")
    except ValueError as e:
        print(f"Could not compute Krippendorffs alpha: {e}")
        return None
