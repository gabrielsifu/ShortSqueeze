import joblib
import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


class DataScientist:
    def __init__(self, sett):
        self.sett = sett
        self.xy_train = joblib.load('./data/xy/xy_train.joblib')
        self.models = {}

    def fit(self):
        used_indices = set()
        previous_model = None

        # Ensure keys are processed in order
        for key in sorted(self.xy_train.keys()):
            df = self.xy_train[key]

            # Separate features and target
            x = df.drop(columns=['y'])
            y = df['y']

            # Get new indices not used in previous training
            new_indices = df.index.difference(used_indices)
            used_indices.update(df.index)

            x_new = x.loc[new_indices]
            y_new = y.loc[new_indices]

            # Compute class weights to balance the classes
            unique_classes = np.unique(y_new)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y_new
            )
            class_weight_dict = dict(zip(unique_classes, class_weights))

            # EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # Neural Network Model
            if previous_model is None:
                # Define a new model
                model_nn = Sequential()
                model_nn.add(Dense(128, activation='relu', input_shape=(x_new.shape[1],)))
                model_nn.add(Dense(128, activation='relu'))
                model_nn.add(Dense(1, activation='sigmoid'))
                model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
            else:
                model_nn = previous_model

            # Train the neural network model with early stopping and class weights
            model_nn.fit(
                x_new,
                y_new,
                epochs=10,
                batch_size=10000,
                validation_split=0.2,
                callbacks=[early_stopping],
                class_weight=class_weight_dict,
                verbose=1
            )

            # Logistic Regression Model without class weights
            model_lr = LogisticRegression(max_iter=10000)
            model_lr.fit(x, y)

            # Logistic Regression Model with balanced class weights
            model_lr_balanced = LogisticRegression(class_weight='balanced', max_iter=10000)
            model_lr_balanced.fit(x, y)

            # Save the models with the corresponding key
            self.models[key] = {
                'neural_network': model_nn,
                'logistic_regression': model_lr,
                'logistic_regression_balanced': model_lr_balanced
            }
            previous_model = model_nn

        # Save all models to the specified directory
        if not os.path.exists('./data/models/'):
            os.makedirs('./data/models/')

        for key, model_dict in self.models.items():
            # Save neural network model
            model_dict['neural_network'].save(f'./data/models/model_nn_{key}.keras')

            # Save logistic regression model without class weights
            joblib.dump(model_dict['logistic_regression'], f'./data/models/model_lr_{key}.joblib')

            # Save logistic regression model with balanced class weights
            joblib.dump(model_dict['logistic_regression_balanced'], f'./data/models/model_lr_balanced_{key}.joblib')
