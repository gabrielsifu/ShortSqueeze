import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates  # Import matplotlib.dates for date formatting

class Evaluator:
    def __init__(self, sett):
        self.sett = sett
        self.predictions = joblib.load('./data/predictions/predictions.joblib')

    def evaluate(self):
        # Ensure the evaluation directory exists
        os.makedirs('data/evaluation/', exist_ok=True)

        # Concatenate all DataFrames into one
        all_predictions = pd.concat(self.predictions.values(), ignore_index=True)

        # Compute cross-entropy loss for each sample
        epsilon = 1e-15  # To prevent log(0)
        all_predictions['loss'] = -(
            all_predictions['y_true'] * np.log(np.clip(all_predictions['y_pred'], epsilon, 1 - epsilon)) +
            (1 - all_predictions['y_true']) * np.log(np.clip(1 - all_predictions['y_pred'], epsilon, 1 - epsilon))
        )

        # Check if 'time' column exists and is in datetime format
        if 'time' in all_predictions.columns:
            # Ensure 'time' is datetime
            all_predictions['time'] = pd.to_datetime(all_predictions['time'])
            # Sort by time
            all_predictions.sort_values('time', inplace=True)
            # Set 'time' as index
            all_predictions.set_index('time', inplace=True)

            # Compute moving average of loss over a 3-month window
            loss_moving_avg = all_predictions['loss'].rolling(window='90D').mean()

            # Plot moving average of loss over time
            plt.figure(figsize=(10, 6))
            plt.plot(loss_moving_avg.index, loss_moving_avg.values)
            plt.title('Moving Average of Cross-Entropy Loss Over Time (Window: 3 Months)')
            plt.xlabel('Time')
            plt.ylabel('Average Cross-Entropy Loss')
            plt.grid(True)

            # Format the x-axis to show years
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to yearly intervals
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format ticks as years
            plt.gcf().autofmt_xdate()  # Auto-format the x-axis labels for better readability

            plt.tight_layout()
            plt.savefig('data/evaluation/loss_moving_average.png')
            plt.close()
        else:
            # Use integer index if 'time' column does not exist
            all_predictions.reset_index(drop=True, inplace=True)
            # Compute moving average over a window of N data points
            window_size = 252*100  # Adjust window size as needed
            loss_moving_avg = all_predictions['loss'].rolling(window=window_size).mean()

            # Plot moving average of loss over index
            plt.figure(figsize=(10, 6))
            plt.plot(loss_moving_avg.values)
            plt.title(f'Moving Average of Cross-Entropy Loss (Window: {window_size} Samples)')
            plt.ylabel('Average Cross-Entropy Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('data/evaluation/loss_moving_average.png')
            plt.close()

        # Prepare data for histograms
        y_pred_0 = all_predictions[all_predictions['y_true'] == 0]['y_pred']
        y_pred_1 = all_predictions[all_predictions['y_true'] == 1]['y_pred']

        # Plot histograms on dual axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'black'
        ax1.hist(y_pred_0, bins=50, alpha=1, label='y_true = 0', color=color1)
        ax1.set_xlabel('Predicted Value (y_pred)')
        ax1.set_ylabel('Frequency (y_true = 0)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  # Create a twin y-axis

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

        print("Evaluation completed and results saved in 'data/evaluation/' directory.")
