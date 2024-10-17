import joblib
import os
import numpy as np
from sklearn.utils import class_weight
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

            if previous_model is None:
                # Define a new model
                model = Sequential()
                model.add(Dense(128, activation='relu', input_shape=(x_new.shape[1],)))
                model.add(Dense(128, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
            else:
                model = previous_model

            # Train the model with early stopping and class weights
            model.fit(
                x_new,
                y_new,
                epochs=20,
                batch_size=10000,
                validation_split=0.2,
                callbacks=[early_stopping],
                class_weight=class_weight_dict,
                verbose=1
            )

            # Save the model with the corresponding key
            self.models[key] = model
            previous_model = model

        # Save all models to the specified directory
        if not os.path.exists('./data/models/'):
            os.makedirs('./data/models/')

        for key, model in self.models.items():
            model.save(f'./data/models/model_{key}.keras')
