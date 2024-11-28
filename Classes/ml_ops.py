import os
import joblib
import pandas as pd
import numpy as np
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
                # Logistic Regression Models and Linear Regression
                if filename.startswith('model_lr_balanced_'):
                    key = filename.replace('model_lr_balanced_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['logistic_regression_balanced'] = joblib.load(os.path.join(model_dir, filename))
                elif filename.startswith('model_lr_') and not filename.startswith('model_lr_balanced_'):
                    key = filename.replace('model_lr_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['logistic_regression'] = joblib.load(os.path.join(model_dir, filename))
                elif filename.startswith('nn_calibrator_'):
                    key = filename.replace('nn_calibrator_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['nn_calibrator'] = joblib.load(os.path.join(model_dir, filename))
                elif filename.startswith('lr_calibrator_'):
                    key = filename.replace('lr_calibrator_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['lr_calibrator'] = joblib.load(os.path.join(model_dir, filename))
                elif filename.startswith('linear_reg_'):
                    key = filename.replace('linear_reg_', '').replace('.joblib', '')
                    if key not in self.models:
                        self.models[key] = {}
                    self.models[key]['linear_regression'] = joblib.load(os.path.join(model_dir, filename))

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
                predictions_df = pd.DataFrame({
                    'y_true': y_true,
                    'log_returns': log_returns
                })

                # Check that all required models are available
                required_models = [
                    'neural_network', 'logistic_regression', 'logistic_regression_balanced',
                    'nn_calibrator', 'lr_calibrator', 'linear_regression'
                ]
                missing_models = [model_name for model_name in required_models if model_name not in model_dict]
                if missing_models:
                    print(f"Missing models for date {date_key}: {missing_models}. Skipping prediction for this date.")
                    continue

                # Calibrated model
                nn_calibrator = model_dict['nn_calibrator']
                lr_calibrator = model_dict['lr_calibrator']

                # Predict with the neural network model
                nn_model = model_dict['neural_network']
                nn_probs = nn_model.predict(x_test).flatten()
                nn_probs_calibrated = nn_calibrator.predict_proba(nn_probs.reshape(-1, 1))[:, 1]
                predictions_df['y_pred_neural_network'] = nn_probs_calibrated

                # Predict with the balanced logistic regression model
                lr_balanced_model = model_dict['logistic_regression_balanced']
                lr_balanced_probs = lr_balanced_model.predict_proba(x_test)[:, 1]
                lr_balanced_probs_calibrated = lr_calibrator.predict_proba(lr_balanced_probs.reshape(-1, 1))[:, 1]
                predictions_df['y_pred_logistic_regression_balanced'] = lr_balanced_probs_calibrated

                # Predict with the unbalanced logistic regression model
                lr_model = model_dict['logistic_regression']
                lr_probs = lr_model.predict_proba(x_test)[:, 1]
                predictions_df['y_pred_logistic_regression'] = lr_probs

                # Predict with the linear regression model
                linear_reg = model_dict['linear_regression']
                x_for_linear = pd.DataFrame({
                    'Spread': x_test['Spread'],  # Assuming 'Spread' exists in x_test
                    'NN_Predictions': nn_probs
                })
                linear_preds = linear_reg.predict(x_for_linear)
                predictions_df['y_pred_linear_regression'] = linear_preds

                # Store predictions for the current date key
                predictions[date_key] = predictions_df
            else:
                print(f"No model found for date {date_key}. Skipping prediction for this date.")

        # Save the predictions dictionary using joblib
        joblib.dump(predictions, os.path.join(predictions_dir, 'predictions.joblib'))
