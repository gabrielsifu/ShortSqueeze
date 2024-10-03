import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:
    def __init__(self, sett):
        self.sett = sett
        self.xy_train = joblib.load('./data/xy_full/xy_train.joblib')
        self.xy_test = joblib.load('./data/xy_full/xy_test.joblib')
        self.stats_dict = joblib.load('./data/xy_full/stats_dict.joblib')
        self.feature_importances_ = None  # Changed to store a single Series

    def select_features(self):
        print("Starting feature selection...")
        self.feature_importance()
        self.drop_high_correlated()
        self.feature_importance()
        self.drop_lower_importance()
        self.save_data()
        print("Feature selection completed.")

    def feature_importance(self):
        print("Calculating feature importance using Random Forest...")
        # Use only the first key to calculate feature importance
        keys_list = list(self.xy_train.keys())
        first_key = keys_list[0]
        print(f"Processing dataset: {first_key}")
        df_train = self.xy_train[first_key]
        x = df_train.drop(columns=['y'])
        y = df_train['y']
        clf = RandomForestClassifier(
            n_estimators=10,  # TODO: Change to 100
            criterion='entropy',
            max_depth=10,
            min_samples_leaf=0.01,
            max_features='sqrt',
            n_jobs=-1,
            verbose=0,
            random_state=42)
        clf.fit(x, y)
        importances = clf.feature_importances_
        self.feature_importances_ = pd.Series(importances, index=x.columns)
        print("Feature importance calculation completed.")

    def drop_high_correlated(self):
        print("Dropping highly correlated features based on the first dataset...")
        # Use only the first key to determine correlated features
        keys_list = list(self.xy_train.keys())
        first_key = keys_list[0]
        df_train = self.xy_train[first_key]
        x = df_train.drop(columns=['y'])
        corr_matrix = x.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for column in upper.columns:
            correlated_features = upper.index[upper[column] >= 0.9].tolist()
            for row in correlated_features:
                imp_col = self.feature_importances_[column]
                imp_row = self.feature_importances_[row]
                if imp_col < imp_row:
                    to_drop.add(column)
                else:
                    to_drop.add(row)
        print(f"Dropping features: {to_drop}")
        # Drop the features from all datasets
        for key in self.xy_train:
            df_train = self.xy_train[key]
            df_train = df_train.drop(columns=list(to_drop), errors='ignore')
            self.xy_train[key] = df_train
            if key in self.xy_test:
                df_test = self.xy_test[key]
                df_test = df_test.drop(columns=list(to_drop), errors='ignore')
                self.xy_test[key] = df_test
        print("Highly correlated features dropped.")

    def drop_lower_importance(self):
        print("Dropping less important features based on the first dataset...")
        # Use only the first key to determine less important features
        keys_list = list(self.xy_train.keys())
        first_key = keys_list[0]
        df_train = self.xy_train[first_key]
        x_columns = df_train.drop(columns=['y']).columns
        importances = self.feature_importances_.reindex(x_columns)
        importances = importances / importances.sum()
        percentile_10 = importances.quantile(0.1)  # TODO: Parametrize on settings
        low_importance_features = importances[importances <= percentile_10].index.tolist()
        print(f"Dropping features: {low_importance_features}")
        # Drop the features from all datasets
        for key in self.xy_train:
            df_train = self.xy_train[key]
            df_train = df_train.drop(columns=low_importance_features, errors='ignore')
            self.xy_train[key] = df_train
            if key in self.xy_test:
                df_test = self.xy_test[key]
                df_test = df_test.drop(columns=low_importance_features, errors='ignore')
                self.xy_test[key] = df_test
        print("Low importance features dropped.")

    def save_data(self):
        print("Saving updated datasets...")
        joblib.dump(self.xy_train, './data/xy/xy_train.joblib')
        joblib.dump(self.xy_test, './data/xy/xy_test.joblib')
        print("Datasets saved.")
