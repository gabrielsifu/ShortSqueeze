from Config.config import sett
from Classes.data_engineer import DataEngineer
from Classes.feature_engineer import FeatureEngineer
from Classes.feature_selector import FeatureSelector
from Classes.data_scientist import DataScientist
from Classes.ml_ops import MLOps
from Classes.evaluator import Evaluator


if __name__ == '__main__':
    # Extract data, Transform, Clean NaN, and Load
    dc = DataEngineer(sett)
    dc.save_clean_data()

    # Create Features, Normalize, Divide Periods
    fe = FeatureEngineer(sett)
    fe.generate_features()

    # Select Features
    fs = FeatureSelector(sett)
    fs.select_features()

    # Model Trainer
    ds = DataScientist(sett)
    ds.fit()

    # Machine Learning Operations
    mlo = MLOps(sett)
    mlo.predict()

    # Evaluator
    e = Evaluator
    e.evaluate()









