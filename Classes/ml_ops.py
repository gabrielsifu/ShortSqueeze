import os
import joblib
import pandas as pd
from tensorflow.keras.models import load_model


class MLOps:
    def __init__(self, sett):
        self.sett = sett
        self.xy_test = joblib.load('./data/xy/xy_test.joblib')
        self.models = {}

        # Load all models from the specified directory
        model_dir = './data/models/'
        for filename in os.listdir(model_dir):
            # Extract the key (date) and model type from the filename
            if filename.endswith('.keras'):
                # Neural Network Model
                if filename.startswith('model_nn_'):
                    key = filename.replace('model_nn_', '').replace('.keras', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['neural_network'] = load_model(os.path.join(model_dir, filename))
            elif filename.endswith('.joblib'):
                # Logistic Regression Models
                if filename.startswith('model_lr_balanced_'):
                    key = filename.replace('model_lr_balanced_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['logistic_regression_balanced'] = joblib.load(os.path.join(model_dir, filename))
                elif filename.startswith('model_lr_'):
                    key = filename.replace('model_lr_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['logistic_regression'] = joblib.load(os.path.join(model_dir, filename))

    def predict(self):
        predictions = {}

        # Ensure the predictions directory exists
        predictions_dir = './data/predictions/'
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)

        for date_key, data in self.xy_test.items():
            # Check if there's a model for this date
            if date_key in self.models:
                model_dict = self.models[date_key]

                # Separate features and target
                x_test = data.drop(columns=['y', 'LogReturns'])
                y_true = data['y']
                log_returns = data['LogReturns']

                # Initialize a DataFrame to store predictions
                predictions_df = pd.DataFrame({'y_true': y_true, 'log_returns': log_returns})

                for model_name, model in model_dict.items():
                    if model_name == 'neural_network':
                        # Neural Network Prediction
                        y_pred = model.predict(x_test)
                        y_pred = y_pred.flatten()
                    else:
                        # Logistic Regression Prediction
                        y_pred = model.predict_proba(x_test)[:, 1]

                    # Add predictions to the DataFrame
                    predictions_df[f'y_pred_{model_name}'] = y_pred

                # Store predictions for the current date key
                predictions[date_key] = predictions_df
            else:
                print(f"No model found for date {date_key}. Skipping prediction for this date.")

        # Save the predictions dictionary using joblib
        joblib.dump(predictions, os.path.join(predictions_dir, 'predictions.joblib'))
