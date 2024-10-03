import joblib as joblib
import pandas as pd
import warnings
import numpy as np


class FeatureEngineer:
    def __init__(self, sett):
        self.sett = sett
        self.data = pd.read_csv('./data/clean_data/data.csv', index_col=['DATE_REF', 'TRADINGITEM_ID'], sep=';')
        date_ref = self.data.index.get_level_values('DATE_REF')  # Extraindo o nível DATE_REF
        tradingitem_id = self.data.index.get_level_values('TRADINGITEM_ID')  # Extraindo o nível TRADINGITEM_ID
        # Converter o nível DATE_REF para datetime
        date_ref = pd.to_datetime(date_ref, errors='coerce')  # Converter para datetime
        # Recriar o MultiIndex com o nível DATE_REF convertido
        self.data.index = pd.MultiIndex.from_arrays([date_ref, tradingitem_id], names=['DATE_REF', 'TRADINGITEM_ID'])
        self.xy_train = None
        self.xy_test = None
        self.stats_dict = None

    def feature_engineering(self):
        print("feature_engineering")
        self.generate_features()
        self.split_periods()
        self.normalize()
        self.save()

        print("end")

    @staticmethod
    def calculate_rolling(df, columns, window_sizes, operations):
        """
        Applies a rolling window on specific columns of a DataFrame with multi-level indices.

        Parameters:
        - df: DataFrame with 'DATE_REF' and 'TRADINGITEM_ID' as indices.
        - columns: List of columns on which to apply the rolling operations.
        - window_sizes: List of rolling window sizes.
        - operations: List of operations to be applied ('max', 'min', 'std', 'sum', 'mean', 'skew', 'median', 'kurt').

        Returns:
        - DataFrame with new calculated columns.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

            # Check if the required indices are in the DataFrame
            if not {'DATE_REF', 'TRADINGITEM_ID'}.issubset(df.index.names):
                raise ValueError("The DataFrame must have 'DATE_REF' and 'TRADINGITEM_ID' as indices.")

            # Dictionary to map available rolling functions
            operations_map = {
                'max': 'max',
                'min': 'min',
                'std': 'std',
                'sum': 'sum',
                'mean': 'mean',
                'skew': 'skew',
                'median': 'median',
                'kurt': 'kurt'
            }

            # Check if all provided operations are valid
            for operation in operations:
                if operation not in operations_map:
                    raise ValueError(f"Invalid operation: {operation}. Choose from: {', '.join(operations_map.keys())}")

            # Initialize a DataFrame to store the new calculated columns
            df_result = df.copy()

            # Apply the operations for each column and each window size
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"Column {col} is not present in the DataFrame.")

                for window_size in window_sizes:
                    for operation in operations:
                        # Name for the new column
                        new_col_name = f"{col}_{operation}_rolling_{window_size}"

                        # Apply the correct rolling operation based on the operation type
                        if operation in ['max', 'min', 'std', 'sum', 'mean']:
                            df_result[new_col_name] = (df.groupby(['TRADINGITEM_ID'])[col]
                                                       .rolling(window=window_size,
                                                                min_periods=int(window_size/2))
                                                       .agg(operations_map[operation])
                                                       .reset_index(level=[0], drop=True))
                        elif operation == 'skew':
                            df_result[new_col_name] = (df.groupby(['TRADINGITEM_ID'])[col]
                                                       .rolling(window=window_size,
                                                                min_periods=int(window_size/2))
                                                       .skew()
                                                       .reset_index(level=[0], drop=True))
                        elif operation == 'median':
                            df_result[new_col_name] = (df.groupby(['TRADINGITEM_ID'])[col]
                                                       .rolling(window=window_size,
                                                                min_periods=int(window_size/2))
                                                       .median()
                                                       .reset_index(level=[0], drop=True))
                        elif operation == 'kurt':
                            df_result[new_col_name] = (df.groupby(['TRADINGITEM_ID'])[col]
                                                       .rolling(window=window_size,
                                                                min_periods=int(window_size/2))
                                                       .kurt()
                                                       .reset_index(level=[0], drop=True))

            return df_result

    def generate_features(self):
        print("generate_features")
        data = self.data.copy()
        data['ContratosAluguel'] = data[
            ['CONTRACT_COUNT/BALCAO', 'CONTRACT_COUNT/Eletronico D+0', 'CONTRACT_COUNT/Eletronico D+1']].sum(axis=1)
        data['SharesAluguel'] = data[
            ['SHARE_COUNT/BALCAO', 'SHARE_COUNT/Eletronico D+0', 'SHARE_COUNT/Eletronico D+1']].sum(axis=1)
        data['PctSharesAluguel'] = data['SharesAluguel']/data['SHARES_OUTSTANDING']
        data['PctSharesAluguelNet'] = data['SharesAluguel'] / (data['SHARES_OUTSTANDING']*data['FREE_FLOAT'])
        data['VolumeAluguel'] = data[
            ['VOLUME/BALCAO', 'VOLUME/Eletronico D+0', 'VOLUME/Eletronico D+1']].sum(axis=1)

        data = self.calculate_rolling(
            data, [
                'LogReturns', 'VolumeTraded',
                'ContratosAluguel', 'SharesAluguel', 'PctSharesAluguel', 'PctSharesAluguelNet', 'VolumeAluguel',
                'DONOR_MIN/BALCAO', 'DONOR_MIN/Eletronico D+0', 'DONOR_MIN/Eletronico D+1',
                'DONOR_MEAN/BALCAO', 'DONOR_MEAN/Eletronico D+0', 'DONOR_MEAN/Eletronico D+1',
                'DONOR_MAX/BALCAO', 'DONOR_MAX/Eletronico D+0', 'DONOR_MAX/Eletronico D+1',
                'TAKER_MIN/BALCAO', 'TAKER_MIN/Eletronico D+0', 'TAKER_MIN/Eletronico D+1',
                'TAKER_MEAN/BALCAO', 'TAKER_MEAN/Eletronico D+0', 'TAKER_MEAN/Eletronico D+1',
                'TAKER_MAX/BALCAO', 'TAKER_MAX/Eletronico D+0', 'TAKER_MAX/Eletronico D+1'
            ],
            [7, 14, 30, 90, 180, 360],
            ['mean', 'median', 'std', 'skew', 'kurt', 'max']
        )
        # TODO: Why full collumns with nan?
        # TODO: Review fill nan
        data = data.dropna(axis=1, how='all')
        grouped_median = data.groupby(level='DATE_REF').median().ffill()
        data.fillna(grouped_median, inplace=True)
        data = data.loc[:, data.std() > 0]
        data.fillna(0, inplace=True)
        data['y'] = (data['LogReturns'] > 0.1).astype(int)
        data.iloc[:, :-1] = data.iloc[:, :-1].shift(-2)
        data.dropna(how='any', inplace=True)
        self.data = data
        print("End")

    @staticmethod
    def create_expanding_windows(df, first_test_year):
        # Assuming the index is a MultiIndex, adjust if necessary
        start_year = df.index[0][0].year
        last_year = df.index[-1][0].year
        train_windows = {}
        test_windows = {}
        # Iterate from the start year to the last available year in the DataFrame
        for end_year in range(first_test_year - 1, last_year):
            window_name = f"{start_year}_{end_year}"
            # Filtering the training window: Convert start_year to Timestamp
            filtered_df_train = df[(df.index.get_level_values(0) >= pd.Timestamp(f'{start_year}-01-01')) &
                                   (df.index.get_level_values(0) <= pd.Timestamp(f'{end_year}-12-31'))]
            train_windows[window_name] = filtered_df_train
            # Filtering the testing window
            filtered_df_test = df[(df.index.get_level_values(0) > pd.Timestamp(f'{end_year}-12-31')) &
                                  (df.index.get_level_values(0) <= pd.Timestamp(f'{end_year + 1}-12-31'))]
            test_windows[window_name] = filtered_df_test
        return train_windows, test_windows

    def split_periods(self):
        print("split_periods")
        first_test_year = 2015  # TODO: Settings
        self.xy_train, self.xy_test = self.create_expanding_windows(self.data, first_test_year)
        print("End")

    @staticmethod
    def truncate_values(series):
        return series.clip(lower=-1, upper=1)

    def normalize(self):  # TODO: Deal with performance warnings
        data_train = self.xy_train.copy()
        data_test = self.xy_test.copy()

        stats_dict = {}
        processed_train = {}
        processed_test = {}

        # Iterate over each key (DataFrame) in the dictionary
        for key in data_train.keys():
            df_train = data_train[key]  # Training DataFrame
            df_test = data_test[key]  # Test DataFrame

            # DataFrames to store processed data
            df_train_processed = pd.DataFrame(index=df_train.index)
            df_test_processed = pd.DataFrame(index=df_test.index)

            # DataFrame to store statistics
            stats = pd.DataFrame(columns=['Percentile 1', 'Percentile 99', 'Mean', 'Standard Deviation'])

            # Iterate over each column (variable) in the DataFrame
            for col in df_train.columns:
                if col == 'y':
                    df_train_processed[col] = df_train['y']
                    df_test_processed[col] = df_test['y']
                    continue
                # Calculate 1st and 99th percentiles for the training set
                p1 = np.percentile(df_train[col], 1)  # TODO: Settings
                p99 = np.percentile(df_train[col], 99)  # TODO: Settings

                # Winsorization: clip values below p1 to p1, and values above p99 to p99
                winsorized_train = np.clip(df_train[col], p1, p99)
                winsorized_test = np.clip(df_test[col], p1, p99)

                # Calculate the mean and standard deviation for the training set after winsorization
                mean = winsorized_train.mean()
                std = winsorized_train.std()

                if std == 0:
                    normalized_train = (winsorized_train - mean)
                    normalized_test = (winsorized_test - mean)
                else:
                    # Normalize: subtract the mean and divide by three times the standard deviation
                    normalized_train = (winsorized_train - mean) / (3 * std)
                    normalized_test = (winsorized_test - mean) / (3 * std)

                # Truncate values between -1 and 1
                truncated_train = self.truncate_values(normalized_train)
                truncated_test = self.truncate_values(normalized_test)

                # Store the processed data in the corresponding DataFrame
                df_train_processed[col] = truncated_train
                df_test_processed[col] = truncated_test

                # Store the statistics for the current column
                stats.loc[col] = [p1, p99, mean, std]

            # Store the processed DataFrames and statistics
            processed_train[key] = df_train_processed
            processed_test[key] = df_test_processed
            stats_dict[key] = stats

        self.xy_train = processed_train
        self.xy_test = processed_test
        self.stats_dict = stats_dict

    def save(self):
        joblib.dump(self.xy_train, './data/xy_full/xy_train.joblib')
        joblib.dump(self.xy_test, './data/xy_full/xy_test.joblib')
        joblib.dump(self.stats_dict, './data/xy_full/stats_dict.joblib')
        # TODO: Why nan?
