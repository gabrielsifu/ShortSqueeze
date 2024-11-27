import pandas as pd
import numpy as np


class DataEngineer:
    def __init__(self, sett):
        self.sett = sett
        self.data = None

    def save_clean_data(self):
        print("save_clean_data")
        self.get_data()
        self.format_data()
        self.fill_nan()
        self.save_data()
        print("end")

    @staticmethod
    def get_data():
        concept_ref = pd.read_csv('./data/raw_data/CONCEPT_REF.csv')
        indicator_ref = pd.read_csv('./data/raw_data/INDICATOR_REF.csv')
        borrowrates = pd.read_csv('./data/raw_data/BORROWRATES.csv')
        concept = pd.read_csv('./data/raw_data/CONCEPT.csv',
                              names=['CONCEPT_ID', 'TRADINGITEM_ID', 'DATE_REF', 'CONCEPT_VALUE'])
        indicator = pd.read_csv('./data/raw_data/INDICATOR.csv')
        marketdata = pd.read_csv('./data/raw_data/MARKETDATA.csv')
        tradingitem = pd.read_csv('./data/raw_data/TRADINGITEM.csv')
        return concept_ref, indicator_ref, borrowrates, concept, indicator, marketdata, tradingitem

    def format_data(self):
        # Loading Data
        concept_ref, indicator_ref, borrowrates, concept, indicator, marketdata, tradingitem = self.get_data()

        # Joins
        # Concept
        concept_wide = pd.merge(concept, concept_ref, on='CONCEPT_ID', how='inner')
        concept_wide.drop(columns=['CONCEPT_ID'], inplace=True)
        concept_wide = concept_wide.pivot(
            index=['DATE_REF', 'TRADINGITEM_ID'], columns='CONCEPT_NAME', values='CONCEPT_VALUE')
        # Indicator
        indicator_wide = pd.merge(indicator, indicator_ref, on='INDICATOR_ID', how='inner')
        indicator_wide.drop(columns=['INDICATOR_ID'], inplace=True)
        indicator_wide = indicator_wide.pivot(
            index=['DATE_REF', 'TRADINGITEM_ID'], columns='INDICATOR_NAME', values='INDICATOR_VALUE')
        # Borrow Rates
        borrowrates.loc[borrowrates.loc[:, 'MARKET'] == 'Balcao', 'MARKET'] = 'BALCAO'
        borrowrates_wide = borrowrates.pivot(
            index=['DATE_REF', 'TRADINGITEM_ID'], columns='MARKET',
            values=['CONTRACT_COUNT', 'SHARE_COUNT', 'VOLUME',
                    'DONOR_MIN', 'DONOR_MEAN', 'DONOR_MAX', 'TAKER_MIN', 'TAKER_MEAN', 'TAKER_MAX'])
        # Market Data
        marketdata_wide = marketdata.set_index(['DATE_REF', 'TRADINGITEM_ID'])
        # Concatening All
        all_wide = pd.concat([concept_wide, indicator_wide, borrowrates_wide, marketdata_wide], axis=1, join='outer')
        all_wide.columns = ['{}/{}'.format(col[0], col[1])
                            if isinstance(col, tuple)
                            else col for col in all_wide.columns]
        if self.sett.DataEngineer.sample:
            all_wide = all_wide.sample(frac=self.sett.DataEngineer.sample)
            all_wide = all_wide.sort_index()
        self.data = all_wide

    def fill_nan(self):
        df = self.data.copy()
        df['VolumeTraded'] = df['ASSETS_TRADED'] * df['PRICE_CLOSE']
        df = df[~pd.to_datetime(df.index.get_level_values('DATE_REF')).weekday.isin([5, 6])]
        zero_fill_cols = ['Dividend', 'LogReturns', 'Returns', 'VolumeTraded', 'AccRet_20', 'AccRet_250_20',
                          'DividendYield', 'Volume_252', 'ASSETS_TRADED',
                          'CONTRACT_COUNT/BALCAO', 'CONTRACT_COUNT/Eletronico D+0', 'CONTRACT_COUNT/Eletronico D+1',
                          'SHARE_COUNT/BALCAO', 'SHARE_COUNT/Eletronico D+0', 'SHARE_COUNT/Eletronico D+1',
                          'VOLUME/BALCAO', 'VOLUME/Eletronico D+0', 'VOLUME/Eletronico D+1',
                          'DONOR_MIN/BALCAO', 'DONOR_MIN/Eletronico D+0', 'DONOR_MIN/Eletronico D+1',
                          'DONOR_MEAN/BALCAO', 'DONOR_MEAN/Eletronico D+0', 'DONOR_MEAN/Eletronico D+1',
                          'DONOR_MAX/BALCAO', 'DONOR_MAX/Eletronico D+0', 'DONOR_MAX/Eletronico D+1',
                          'TAKER_MIN/BALCAO', 'TAKER_MIN/Eletronico D+0', 'TAKER_MIN/Eletronico D+1',
                          'TAKER_MEAN/BALCAO', 'TAKER_MEAN/Eletronico D+0', 'TAKER_MEAN/Eletronico D+1',
                          'TAKER_MAX/BALCAO', 'TAKER_MAX/Eletronico D+0', 'TAKER_MAX/Eletronico D+1']
        df.loc[:, zero_fill_cols] = df.loc[:, zero_fill_cols].fillna(0)
        df['VolumeAluguel'] = df['VOLUME/BALCAO'] + df['VOLUME/Eletronico D+0'] + df['VOLUME/Eletronico D+1']

        for period in [5, 21, 63]:
            df['VolumeAluguel'] = df.groupby(level=['TRADINGITEM_ID']).rolling(
                period).median()['VolumeAluguel'].reset_index(level=0, drop=True)
            df = df[df['VolumeAluguel'] > 0]
            df['Volume'] = df.groupby(level=['TRADINGITEM_ID']).rolling(
                period).median()['VolumeTraded'].reset_index(level=0, drop=True)
            df = df[df['Volume'] > 3000000]

        df.drop(['VolumeAluguel', 'Volume'], axis=1, inplace=True)

        forward_fill_cols = df.columns
        df = df.groupby(level='TRADINGITEM_ID')[forward_fill_cols].ffill()
        grouped_median = df.groupby(level='DATE_REF').median().ffill()
        df = df.fillna(grouped_median)
        df = df.dropna()
        self.data = df

    def save_data(self):
        self.data.to_csv('./data/clean_data/data.csv', sep=';', index=True, encoding='utf-8')
