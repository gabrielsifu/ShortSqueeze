class FeatureSelector:
    def __init__(self, sett):
        self.sett = sett

    def select_features(self):
        print("select_features")
        self.feature_importance()
        self.drop_high_correlated()
        self.feature_importance()
        self.drop_lower_importance()
        print("end")

    def feature_importance(self):
        pass

    def drop_high_correlated(self):
        pass

    def drop_lower_importance(self):
        pass

