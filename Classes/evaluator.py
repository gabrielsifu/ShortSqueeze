import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef
)


class Evaluator:
    def __init__(self, sett):
        self.sett = sett
        self.predictions = joblib.load('./data/predictions/predictions.joblib')

    def ensure_evaluation_directory(self):
        """Ensure the evaluation directory exists."""
        os.makedirs('data/evaluation/', exist_ok=True)

    def concatenate_predictions(self):
        """Concatenate all prediction DataFrames into one."""
        all_predictions = pd.concat(
            self.predictions.values(),
            keys=self.predictions.keys()
        ).reset_index(level=[0, 2], drop=True)
        all_predictions.reset_index(inplace=True)
        return all_predictions

    def compute_cross_entropy_loss(self, all_predictions):
        """Compute cross-entropy loss for each sample."""
        epsilon = 1e-15  # To prevent log(0)
        all_predictions['loss'] = -(
            all_predictions['y_true'] * np.log(np.clip(all_predictions['y_pred'], epsilon, 1 - epsilon)) +
            (1 - all_predictions['y_true']) * np.log(np.clip(1 - all_predictions['y_pred'], epsilon, 1 - epsilon))
        )
        return all_predictions

    def prepare_data(self, all_predictions):
        """Prepare the data by ensuring datetime format and sorting."""
        all_predictions['DATE_REF'] = pd.to_datetime(all_predictions['DATE_REF'])
        all_predictions.sort_values('DATE_REF', inplace=True)
        return all_predictions

    def compute_moving_average_loss(self, all_predictions):
        """Compute moving average of loss over a 30-day window."""
        all_predictions_grouped = all_predictions.groupby('DATE_REF').mean()
        loss_moving_avg = all_predictions_grouped['loss'].rolling(window='30D', min_periods=15).mean()
        return loss_moving_avg

    def plot_moving_average_loss(self, loss_moving_avg):
        """Plot and save the moving average of loss over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_moving_avg.index, loss_moving_avg.values)
        plt.title('Moving Average of Cross-Entropy Loss Over Time (Window: 1 Month)')
        plt.xlabel('DATE_REF')
        plt.ylabel('Average Cross-Entropy Loss')
        plt.grid(True)

        # Format the x-axis to show years
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.savefig('data/evaluation/loss_moving_average.png')
        plt.close()

    def prepare_histogram_data(self, all_predictions):
        """Separate predictions based on true labels for histogram plotting."""
        y_pred_0 = all_predictions[all_predictions['y_true'] == 0]['y_pred']
        y_pred_1 = all_predictions[all_predictions['y_true'] == 1]['y_pred']
        return y_pred_0, y_pred_1

    def plot_histograms(self, y_pred_0, y_pred_1):
        """Plot and save histograms of predicted values."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'black'
        ax1.hist(y_pred_0, bins=50, alpha=1, label='y_true = 0', color=color1)
        ax1.set_xlabel('Predicted Value (y_pred)')
        ax1.set_ylabel('Frequency (y_true = 0)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()

        color2 = 'tab:red'
        ax2.hist(y_pred_1, bins=50, alpha=0.5, label='y_true = 1', color=color2)
        ax2.set_ylabel('Frequency (y_true = 1)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        plt.title('Predicted Values by True Label')
        plt.tight_layout()
        plt.savefig('data/evaluation/predicted_values_histogram_dual_axis.png')
        plt.close()

    def compute_confusion_matrix(self, all_predictions):
        """Compute and plot the confusion matrix without using seaborn."""
        # Binarize predictions based on threshold 0.5
        y_true = all_predictions['y_true']
        y_pred_binary = (all_predictions['y_pred'] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)

        # Save confusion matrix as a table
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        cm_df.to_csv('data/evaluation/confusion_matrix.csv')

        # Plot confusion matrix using matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', y=1.1)
        fig.colorbar(cax)

        # Set axis labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Loop over data dimensions and create text annotations.
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, f'{z}', ha='center', va='center', color='red')

        plt.tight_layout()
        plt.savefig('data/evaluation/confusion_matrix.png')
        plt.close()

    def compute_roc_auc(self, all_predictions):
        """Compute ROC AUC curve and its value."""
        y_true = all_predictions['y_true']
        y_pred_prob = all_predictions['y_pred']
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = roc_auc_score(y_true, y_pred_prob)

        # Save ROC AUC value
        with open('data/evaluation/roc_auc_value.txt', 'w') as f:
            f.write(f'ROC AUC: {roc_auc}')

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('data/evaluation/roc_curve.png')
        plt.close()

    def compute_classification_metrics(self, all_predictions):
        """Compute precision, recall, F1, and Matthews correlation coefficient."""
        y_true = all_predictions['y_true']
        y_pred_binary = (all_predictions['y_pred'] >= 0.5).astype(int)

        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_true, y_pred_binary)

        # Save metrics to a text file
        with open('data/evaluation/classification_metrics.txt', 'w') as f:
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'Matthews Correlation Coefficient: {mcc}\n')

    def compute_lift_curve(self, all_predictions):
        """Compute and plot the lift curve."""
        # Sort predictions by predicted probability in descending order
        data = all_predictions.copy()
        data.sort_values('y_pred', ascending=False, inplace=True)

        # Calculate cumulative true positives
        data['cum_tp'] = data['y_true'].cumsum()

        # Total positive cases
        total_positives = data['y_true'].sum()
        total_cases = len(data)

        # Calculate lift
        data['cum_pct_cases'] = np.arange(1, total_cases + 1) / total_cases
        data['cum_pct_positives'] = data['cum_tp'] / total_positives
        data['lift'] = data['cum_pct_positives'] / data['cum_pct_cases']

        # Plot lift curve
        plt.figure(figsize=(8, 6))
        plt.plot(data['cum_pct_cases'], data['lift'], label='Lift Curve')
        plt.xlabel('Percentage of Sample')
        plt.ylabel('Lift')
        plt.title('Lift Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('data/evaluation/lift_curve.png')
        plt.close()

        # Save lift data to CSV
        lift_data = data[['cum_pct_cases', 'lift']]
        lift_data.to_csv('data/evaluation/lift_data.csv', index=False)

    def evaluate(self):
        """Run the full evaluation process."""
        self.ensure_evaluation_directory()
        all_predictions = self.concatenate_predictions()
        all_predictions = self.compute_cross_entropy_loss(all_predictions)
        all_predictions = self.prepare_data(all_predictions)
        loss_moving_avg = self.compute_moving_average_loss(all_predictions)
        self.plot_moving_average_loss(loss_moving_avg)
        y_pred_0, y_pred_1 = self.prepare_histogram_data(all_predictions)
        self.plot_histograms(y_pred_0, y_pred_1)

        # Compute and plot confusion matrix without seaborn
        self.compute_confusion_matrix(all_predictions)
        # Compute ROC AUC and plot ROC curve
        self.compute_roc_auc(all_predictions)
        # Compute classification metrics
        self.compute_classification_metrics(all_predictions)
        # Compute and plot lift curve
        self.compute_lift_curve(all_predictions)

        print("Evaluation completed and results saved in 'data/evaluation/' directory.")
