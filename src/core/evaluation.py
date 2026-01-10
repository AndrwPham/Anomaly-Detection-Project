import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix

class AD_Evaluation:
    def __init__(self, y_true, y_score, plot_dir: str):
        self.y_true = np.array(y_true, dtype=np.int32)
        self.y_score = np.array(y_score, dtype=np.float32)
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

    def evaluate(self):
        metrics = {}

        # ROC / AUROC
        fpr, tpr, roc_th = roc_curve(self.y_true, self.y_score)
        metrics["AUROC"] = float(roc_auc_score(self.y_true, self.y_score))

        # Youden's J = tpr - fpr
        j = tpr - fpr
        best = int(np.argmax(j))
        metrics["Youden's J threshold"] = float(roc_th[best])
        metrics["Youden's J"] = float(j[best])

        # PR / AP
        prec, rec, pr_th = precision_recall_curve(self.y_true, self.y_score)
        metrics["Average Precision"] = float(average_precision_score(self.y_true, self.y_score))

        # Confusion at best threshold
        thr = metrics["Youden's J threshold"]
        y_pred = (self.y_score > thr).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        metrics.update({
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        })

        # Plots
        self._plot_roc(fpr, tpr)
        self._plot_pr(rec, prec)

        # Save metrics
        with open(os.path.join(self.plot_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        return metrics

    def _plot_roc(self, fpr, tpr):
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "roc_curve.png"), dpi=150)
        plt.close()

    def _plot_pr(self, rec, prec):
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "pr_curve.png"), dpi=150)
        plt.close()
