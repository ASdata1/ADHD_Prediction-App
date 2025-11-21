import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
    classification_report
)


class ModelEvaluator:
    """
    Evaluates ML models using:
        - macro precision
        - macro recall
        - macro F1
        - macro ROC-AUC
        - confusion matrix
        - ROC curve plot
        - confusion matrix heatmap
    All plots are saved to the results directory.
    """

    def __init__(self, results_path):
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true),
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix Heatmap")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return cm

    def plot_roc(self, y_true, y_proba, save_path):
        """
        Supports multiclass ROC using one-vs-rest (OvR).
        """
        plt.figure(figsize=(7, 6))

        # Multiclass ROC
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(y_proba.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_true == i, y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.3f})")

        else:
            # Binary ROC
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc_val:.3f})")

        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def evaluate(self, model, X_val, y_val):
        preds = model.predict(X_val)

        # Probabilities for ROC-AUC
        try:
            proba = model.predict_proba(X_val)
            roc_auc = roc_auc_score(y_val, proba, multi_class="ovr")
        except Exception:
            proba = None
            roc_auc = None

        # ------------------------
        # SAVE CONFUSION MATRIX
        # ------------------------
        cm_path = os.path.join(self.results_path, "confusion_matrix.png")
        cm = self.plot_confusion_matrix(y_val, preds, cm_path)

        # ------------------------
        # SAVE ROC CURVE
        # ------------------------
        if proba is not None:
            roc_path = os.path.join(self.results_path, "roc_curve.png")
            self.plot_roc(y_val, proba, roc_path)
        else:
            roc_path = None

        # ------------------------
        # RETURN METRICS + PATHS
        # ------------------------
        return {
            "macro_precision": precision_score(y_val, preds, average="macro"),
            "macro_recall": recall_score(y_val, preds, average="macro"),
            "macro_f1": f1_score(y_val, preds, average="macro"),
            "macro_roc_auc": roc_auc,
            "classification_report": classification_report(y_val, preds),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_path": cm_path,
            "roc_curve_path": roc_path,
        }
