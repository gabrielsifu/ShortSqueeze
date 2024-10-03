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
            if filename.endswith('.keras'):
                # Extract the key (date) from the filename
                key = filename.replace('model_', '').replace('.keras', '')
                self.models[key] = load_model(os.path.join(model_dir, filename))

    def predict(self):
        predictions = {}

        # Ensure the predictions directory exists
        predictions_dir = './data/predictions/'
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)

        for date_key, data in self.xy_test.items():
            # Check if there's a model for this date
            if date_key in self.models:
                model = self.models[date_key]

                # Assuming 'y' is the last column in the DataFrame
                x_test = data.drop(columns=['y'])
                y_true = data['y']

                # Make predictions
                y_pred = model.predict(x_test)
                # Flatten y_pred in case it's in a nested array
                y_pred = y_pred.flatten()

                # Store predictions in a DataFrame
                predictions[date_key] = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred': y_pred
                })

            else:
                print(f"No model found for date {date_key}. Skipping prediction for this date.")

        # Save the predictions dictionary using joblib
        joblib.dump(predictions, os.path.join(predictions_dir, 'predictions.joblib'))
