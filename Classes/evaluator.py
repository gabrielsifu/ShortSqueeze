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
        """Compute cross-entropy loss for each sample for each model."""
        epsilon = 1e-15  # To prevent log(0)
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            loss_col = f'loss_{model_col}'
            all_predictions[loss_col] = -(
                all_predictions['y_true'] * np.log(np.clip(all_predictions[model_col], epsilon, 1 - epsilon)) +
                (1 - all_predictions['y_true']) * np.log(np.clip(1 - all_predictions[model_col], epsilon, 1 - epsilon))
            )
        return all_predictions

    def prepare_data(self, all_predictions):
        """Prepare the data by ensuring datetime format and sorting."""
        all_predictions['DATE_REF'] = pd.to_datetime(all_predictions['DATE_REF'])
        all_predictions.sort_values('DATE_REF', inplace=True)
        return all_predictions

    def compute_moving_average_loss(self, all_predictions):
        """Compute moving average of loss over a 30-day window for each model."""
        model_loss_columns = [col for col in all_predictions.columns if col.startswith('loss_y_pred')]
        all_predictions_grouped = all_predictions.groupby('DATE_REF').mean()
        loss_moving_avg = {}

        for loss_col in model_loss_columns:
            loss_moving_avg[loss_col] = all_predictions_grouped[loss_col].rolling(window='360D', min_periods=180).mean()
        return loss_moving_avg

    def plot_moving_average_loss(self, loss_moving_avg):
        """Plot and save the moving average of loss over time for each model."""
        plt.figure(figsize=(10, 6))
        for loss_col, series in loss_moving_avg.items():
            if loss_col == 'loss_y_pred_logistic_regression':
                continue
            plt.plot(series.index, series.values, label=loss_col.replace('loss_y_pred_', ''))

        # plt.title('Moving Average of Cross-Entropy Loss Over Time (Window: 1 Year)')
        plt.title('Média Móvel (1 ano) da Entropia Cruzada')
        # plt.xlabel('DATE_REF')
        plt.xlabel('Ano')
        # plt.ylabel('Average Cross-Entropy Loss')
        plt.ylabel('Entropia Cruzada Média')
        plt.legend()
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
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]
        histograms = {}

        for model_col in model_columns:
            y_pred_0 = all_predictions[all_predictions['y_true'] == 0][model_col]
            y_pred_1 = all_predictions[all_predictions['y_true'] == 1][model_col]
            histograms[model_col] = (y_pred_0, y_pred_1)
        return histograms

    def plot_histograms(self, histograms):
        """Plot and save histograms of predicted values for each model."""
        for model_col, (y_pred_0, y_pred_1) in histograms.items():
            fig, ax1 = plt.subplots(figsize=(10, 6))

            color1 = 'black'
            ax1.hist(y_pred_0, bins=50, alpha=1, label='y_true = 0', color=color1)
            # ax1.set_xlabel('Predicted Value (y_pred)')
            ax1.set_xlabel('Valor Previsto (y_pred)')
            # ax1.set_ylabel('Frequency (y_true = 0)', color=color1)
            ax1.set_ylabel('Frequência (y_true = 0)', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)

            ax2 = ax1.twinx()

            color2 = 'tab:red'
            ax2.hist(y_pred_1, bins=50, alpha=0.5, label='y_true = 1', color=color2)
            # ax2.set_ylabel('Frequency (y_true = 1)', color=color2)
            ax2.set_ylabel('Frequência (y_true = 1)', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)

            # Add legends
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

            model_name = model_col.replace('y_pred_', '')
            # plt.title(f'Predicted Values by True Label - {model_name}')
            plt.title(f'Valores preditos pelo rótulo real - {model_name}')
            plt.tight_layout()
            plt.savefig(f'data/evaluation/predicted_values_histogram_dual_axis_{model_name}.png')
            plt.close()

    def compute_confusion_matrix(self, all_predictions):
        """Compute and plot the confusion matrix for each model."""
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            # Binarize predictions based on threshold 0.5
            y_true = all_predictions['y_true']
            y_pred_binary = (all_predictions[model_col] >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)

            # Save confusion matrix as a table
            cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
            model_name = model_col.replace('y_pred_', '')
            cm_df.to_csv(f'data/evaluation/confusion_matrix_{model_name}.csv')

            # Plot confusion matrix using matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}', y=1.1)
            fig.colorbar(cax)

            # Set axis labels
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # Loop over data dimensions and create text annotations.
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, f'{z}', ha='center', va='center', color='red')

            plt.tight_layout()
            plt.savefig(f'data/evaluation/confusion_matrix_{model_name}.png')
            plt.close()

    def compute_roc_auc(self, all_predictions):
        """Compute ROC AUC curve and its value for each model."""
        y_true = all_predictions['y_true']
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            y_pred_prob = all_predictions[model_col]
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
            roc_auc = roc_auc_score(y_true, y_pred_prob)

            model_name = model_col.replace('y_pred_', '')

            # Save ROC AUC value
            with open(f'data/evaluation/roc_auc_value_{model_name}.txt', 'w') as f:
                f.write(f'ROC AUC ({model_name}): {roc_auc}')

            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            # plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
            plt.plot(fpr, tpr, label=f'Curva ROC (área = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            plt.xlabel('Taxa de Falso Positivo')
            # plt.ylabel('True Positive Rate')
            plt.ylabel('Taxa de Verdadeiro Positivo')
            # plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
            plt.title(f'Curva ROC - {model_name}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(f'data/evaluation/roc_curve_{model_name}.png')
            plt.close()

    def compute_classification_metrics(self, all_predictions):
        """Compute precision, recall, F1, and Matthews correlation coefficient for each model."""
        y_true = all_predictions['y_true']
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            y_pred_binary = (all_predictions[model_col] >= 0.5).astype(int)

            precision = precision_score(y_true, y_pred_binary)
            recall = recall_score(y_true, y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary)
            mcc = matthews_corrcoef(y_true, y_pred_binary)

            model_name = model_col.replace('y_pred_', '')

            # Save metrics to a text file
            with open(f'data/evaluation/classification_metrics_{model_name}.txt', 'w') as f:
                f.write(f'Precision: {precision}\n')
                f.write(f'Recall: {recall}\n')
                f.write(f'F1 Score: {f1}\n')
                f.write(f'Matthews Correlation Coefficient: {mcc}\n')

    def compute_lift_curve(self, all_predictions):
        """Compute and plot the lift curve for each model."""
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            # Sort predictions by predicted probability in descending order
            data = all_predictions[['y_true', model_col]].copy()
            data.sort_values(model_col, ascending=False, inplace=True)

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
            plt.title(f'Lift Curve - {model_col.replace("y_pred_", "")}')
            plt.legend(loc='best')
            plt.grid(True)
            plt.savefig(f'data/evaluation/lift_curve_{model_col.replace("y_pred_", "")}.png')
            plt.close()

            # Save lift data to CSV
            lift_data = data[['cum_pct_cases', 'lift']]
            model_name = model_col.replace('y_pred_', '')
            lift_data.to_csv(f'data/evaluation/lift_data_{model_name}.csv', index=False)

    def evaluate(self):
        """Run the full evaluation process."""
        self.ensure_evaluation_directory()
        all_predictions = self.concatenate_predictions()
        all_predictions = self.compute_cross_entropy_loss(all_predictions)
        all_predictions = self.prepare_data(all_predictions)
        loss_moving_avg = self.compute_moving_average_loss(all_predictions)
        self.plot_moving_average_loss(loss_moving_avg)
        histograms = self.prepare_histogram_data(all_predictions)
        self.plot_histograms(histograms)

        # Compute and plot confusion matrix for each model
        self.compute_confusion_matrix(all_predictions)
        # Plot return distribution
        self.plot_return_distribution(all_predictions)
        # Compute ROC AUC and plot ROC curve for each model
        self.compute_roc_auc(all_predictions)
        # Compute classification metrics for each model
        self.compute_classification_metrics(all_predictions)
        # Compute and plot lift curve for each model
        self.compute_lift_curve(all_predictions)

        print("Evaluation completed and results saved in 'data/evaluation/' directory.")

    def plot_return_distribution(self, df):
        # Get the model names by parsing the columns
        pred_cols = [col for col in df.columns if col.startswith('y_pred_')]
        models = [col.replace('y_pred_', '') for col in pred_cols]

        for model in models:
            # Get the predicted labels
            y_pred_col = 'y_pred_' + model

            # Create a column for TP, TN, FP, FN labels
            confusion_label_col = 'confusion_label_' + model
            df[confusion_label_col] = df.apply(
                lambda row: self.confusion_label(row['y_true'], row[y_pred_col]), axis=1)

            # Arrange labels to match the desired quadrant layout
            labels = ['TN', 'FP', 'FN', 'TP']

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            for idx, label in enumerate(labels):
                subset = df[df[confusion_label_col] == label]
                data = subset['log_returns']
                ax = axes[idx]
                ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{model} - {label} ({len(data)})')

                # Calculate statistics
                mean = data.mean()
                std = data.std()
                median = data.median()
                q75 = data.quantile(0.75)
                p90 = data.quantile(0.90)
                p95 = data.quantile(0.95)

                # Plot vertical lines for statistics
                ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {100*mean:.2f}%')
                ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {100*median:.2f}%')
                ax.axvline(q75, color='blue', linestyle='dashed', linewidth=1, label=f'75th percentile: {100*q75:.2f}%')
                ax.axvline(p90, color='cyan', linestyle='dashed', linewidth=1, label=f'90th percentile: {100*p90:.2f}%')
                ax.axvline(p95, color='magenta', linestyle='dashed', linewidth=1, label=f'95th percentile: {100*p95:.2f}%')
                ax.axvline(mean, color='grey', linestyle='dashed', linewidth=0.5, label=f'Std: {100*std:.2f}%')

                ax.legend()
                ax.set_xlabel('Log Returns')
                ax.set_ylabel('Frequency')

            plt.tight_layout()
            plt.savefig(f'data/evaluation/{model}_histograms_returns.png')
            plt.close()

    @staticmethod
    def confusion_label(y_true, y_pred, thresh=0.5):
        if y_true == 1 and y_pred > thresh:
            return 'TP'
        elif y_true == 0 and y_pred <= thresh:
            return 'TN'
        elif y_true == 0 and y_pred > thresh:
            return 'FP'
        elif y_true == 1 and y_pred <= thresh:
            return 'FN'
        else:
            return 'Unknown'
