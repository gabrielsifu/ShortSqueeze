import pandas as pd


class FeatureEngineer:
    def __init__(self, sett):
        self.sett = sett

    def generate_features(self):
        print("generate_features")
        data = pd.read_csv('./data/clean_data/data.csv', index_col=['date', 'tradingitem_id'])

        print("end")





