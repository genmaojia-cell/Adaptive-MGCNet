import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_metrics(y_true, y_probs, threshold=0.5):
    """
    Calculate 6 standard clinical evaluation metrics for ASD classification.
    Directly matches the evaluation suite in run7_large8_with_combat_3.py.
    """
    y_preds = (y_probs > threshold).astype(int)

    acc = accuracy_score(y_true, y_preds)
    pre = precision_score(y_true, y_preds, zero_division=0)
    sen = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)

    # Specificity computation via confusion matrix
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spe = 0.0

    # AUC calculation with multi-class protection
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
    else:
        auc = np.nan

    return {
        'acc': acc,
        'pre': pre,
        'sen': sen,
        'spe': spe,
        'f1': f1,
        'auc': auc
    }