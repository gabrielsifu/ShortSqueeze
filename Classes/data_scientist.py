import joblib
import os
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class DataScientist:
    def __init__(self, sett):
        self.sett = sett
        self.xy_train = joblib.load('./data/xy/xy_train.joblib')
        self.models = {}
        self.x_train_full = pd.DataFrame()
        self.y_train_full = pd.Series(dtype='int')
        self.x_val = pd.DataFrame()
        self.y_val = pd.Series(dtype='int')

    def fit(self):
        used_indices = set()
        model_nn = None  # Initialize neural network model

        # Ensure keys are processed in order
        for key in sorted(self.xy_train.keys()):
            df = self.xy_train[key]

            # Separate features and target
            x = df.drop(columns=['y', 'LogReturns'])
            y = df['y']

            # Get new indices not used in previous training
            new_indices = df.index.difference(used_indices)
            used_indices.update(df.index)

            x_new = x.loc[new_indices]
            y_new = y.loc[new_indices]

            # Split new data into training and validation sets
            x_new_train, x_new_val, y_new_train, y_new_val = train_test_split(
                x_new, y_new, test_size=0.2, random_state=42, stratify=y_new
            )

            # Accumulate full training data for logistic regression models
            self.x_train_full = pd.concat([self.x_train_full, x_new_train])
            self.y_train_full = pd.concat([self.y_train_full, y_new_train])

            # Accumulate validation data
            self.x_val = pd.concat([self.x_val, x_new_val])
            self.y_val = pd.concat([self.y_val, y_new_val])

            # Compute class weights using cumulative training labels
            unique_classes = np.unique(self.y_train_full)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=self.y_train_full
            )
            class_weight_dict = dict(zip(unique_classes, class_weights))

            # Compute sample weights for the validation data
            val_sample_weights = self.y_val.map(class_weight_dict).values

            # EarlyStopping callback monitoring validation AUC
            early_stopping = EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',  # Since we aim to maximize AUC
                restore_best_weights=True
            )

            # Neural Network Model
            if model_nn is None:
                # Define a new model
                model_nn = Sequential([
                    Dense(128, activation='relu', input_shape=(x_new_train.shape[1],)),
                    # Dense(128, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                optimizer = Adam(learning_rate=0.001)
                model_nn.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=[tf.keras.metrics.AUC(name='auc')]
                )

            # Train the neural network model incrementally on new training data
            model_nn.fit(
                x_new_train,
                y_new_train,
                epochs=10000,
                batch_size=1000,
                validation_data=(self.x_val, self.y_val, val_sample_weights),
                callbacks=[early_stopping],
                class_weight=class_weight_dict,
                verbose=1
            )
            nn_probs_train = model_nn.predict(self.x_train_full).flatten()
            nn_calibrator = LogisticRegression(max_iter=10000)
            nn_calibrator.fit(nn_probs_train.reshape(-1, 1), self.y_train_full)

            # Logistic Regression Model without class weights
            model_lr = LogisticRegression(max_iter=10000)
            model_lr.fit(self.x_train_full, self.y_train_full)

            # Logistic Regression Model with balanced class weights
            model_lr_balanced = LogisticRegression(class_weight='balanced', max_iter=10000)
            model_lr_balanced.fit(self.x_train_full, self.y_train_full)
            lr_balanced_probs_train = model_lr_balanced.predict_proba(self.x_train_full)[:, 1]
            lr_calibrator = LogisticRegression(max_iter=10000)
            lr_calibrator.fit(lr_balanced_probs_train.reshape(-1, 1), self.y_train_full)

            # Save the models with the corresponding key
            self.models[key] = {
                'neural_network': model_nn,
                'logistic_regression': model_lr,
                'logistic_regression_balanced': model_lr_balanced,
                'nn_calibrator': nn_calibrator,
                'lr_calibrator': lr_calibrator
            }

        # Save all models to the specified directory
        os.makedirs('./data/models/', exist_ok=True)

        for key, model_dict in self.models.items():
            # Save neural network model
            model_dict['neural_network'].save(f'./data/models/model_nn_{key}.keras')

            # Save logistic regression models
            joblib.dump(model_dict['logistic_regression'], f'./data/models/model_lr_{key}.joblib')
            joblib.dump(model_dict['logistic_regression_balanced'], f'./data/models/model_lr_balanced_{key}.joblib')
            joblib.dump(model_dict['nn_calibrator'], f'./data/models/nn_calibrator_{key}.joblib')
            joblib.dump(model_dict['lr_calibrator'], f'./data/models/lr_calibrator_{key}.joblib')
