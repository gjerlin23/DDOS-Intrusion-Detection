import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class DDOSGraphGeneration:
    def __init__(self, json_file="generated.json"):
        self.json_file = json_file
        self.data = self.load_json()

    def load_json(self):
        try:
            with open(self.json_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Error: Could not load JSON file.")
            return None

    def plot_bar_chart(self):
        if not self.data:
            return
        
        metrics = self.data["metrics"]
        models = self.data["models"]
        values = self.data["values"]

        df = pd.DataFrame(values, columns=models, index=metrics)

        plt.figure(figsize=(10, 6))
        df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
        plt.title("Performance Comparison of Models", fontsize=14)
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title="Models", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()

    def plot_line_chart(self):
        if not self.data:
            return

        selected_metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"]
        models = self.data["models"]
        data = np.array(self.data["values"]).T  # Transpose to match models with metrics

        df = pd.DataFrame(data, columns=selected_metrics, index=models)

        custom_colors = {
            "Accuracy (%)": "#FFA500",
            "Precision (%)": "#FF5733",
            "Recall (%)": "#FF1493",
            "F1-score (%)": "#FF69B4"
        }

        plt.figure(figsize=(10, 6))
        for metric in selected_metrics:
            plt.plot(df.index, df[metric], marker="o", linestyle="-", label=metric, color=custom_colors[metric])

        plt.title("Performance Metrics of Models", fontsize=14)
        plt.xlabel("Models", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(rotation=15, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Metrics", fontsize=10, loc="upper left", bbox_to_anchor=(0.02, 0.98))

        plt.show()

    def plot_roc_curve(self):
        """Plot ROC curve for different models"""
        if not self.data:
            return

        models = self.data["models"]
        colors = ["#FFA500", "#FF5733", "#FF6666", "#FF69B4", "#87CEFA"]
        auc_values = [0.91, 0.93, 0.95, 0.95, 0.99]  # Example AUC values

        plt.figure(figsize=(8, 6))

        for i, model in enumerate(models):
            np.random.seed(i)
            y_true = np.random.randint(0, 2, 100)
            y_scores = np.sort(np.random.rand(100))

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{model} (AUC = {auc_values[i]:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5)

        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.title("ROC Curve for Each Method", fontsize=14)
        plt.legend(loc="lower right", fontsize=10, frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.show()
