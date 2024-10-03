from Config.config import sett
from Classes.data_engineer import DataEngineer
from Classes.feature_engineer import FeatureEngineer
from Classes.feature_selector import FeatureSelector
from Classes.data_scientist import DataScientist
from Classes.ml_ops import MLOps
from Classes.evaluator import Evaluator


if __name__ == '__main__':
    if sett.DataEngineer.execute:
        # Extract data, Transform, Clean NaN, and Load
        dc = DataEngineer(sett)
        dc.save_clean_data()

    if sett.FeatureEngineer.execute:
        # Create Features, Normalize, Divide Periods
        fe = FeatureEngineer(sett)
        fe.feature_engineering()

    if sett.FeatureSelector.execute:
        # Select Features
        fs = FeatureSelector(sett)
        fs.select_features()

    if sett.DataScientist.execute:
        # Model Trainer
        ds = DataScientist(sett)
        ds.fit()

    if sett.MLOps.execute:
        # Machine Learning Operations
        mlo = MLOps(sett)
        mlo.predict()

    if sett.Evaluator.execute:
        # Evaluator
        e = Evaluator(sett)
        e.evaluate()

