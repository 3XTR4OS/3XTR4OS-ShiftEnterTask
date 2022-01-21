import pandas as pd
import numpy as np


class MeanDummyRegressor():
    def predict(self, x):
        return np.mean(x)


class PredictCAO:
    
    def __init__(self, model, feature_engineering):
        self.model = model
        self.feature_engineering = feature_engineering

    def predict(self, charge_chemistry, limestone_consumptions,
                charge_consumptions, coke_consumptions,
                coke_sieving, limestone_cao):

        features = self.__prepare_data(
            charge_chemistry,
            limestone_consumptions,
            charge_consumptions,
            coke_consumptions,
            coke_sieving,
            limestone_cao
        )
        return self.__make_prediction(features.iloc[1])

    def __make_df(self, raw_data):
        df = pd.DataFrame(data=raw_data)
        if df.empty:
            raise ValueError()
        return df

    def __prepare_data(self, charge_chemistry, limestone_consumptions,
                       charge_consumptions, coke_consumptions, coke_sieving,
                       limestone_cao):
        limestone_consumption = self.__make_df(limestone_consumptions)

        charge_consumption = self.__make_df(charge_consumptions)
        charge_consumption.columns = ['DATETIME',
                                      'concentrate_limestone_consumption']

        concentrate_limestone_consumption = self.__make_df(charge_consumptions)

        coke_percent = self.__make_df(coke_consumptions)

        Sieving3mm_fuel = self.__make_df(coke_sieving)

        cao_limestone = self.__make_df(limestone_cao)

        charge = self.__make_df(charge_chemistry)
        params = {
            'limestone_consumption': limestone_consumption,
            'charge_consumption': charge_consumption,
            'concentrate_limestone_consumption': concentrate_limestone_consumption,
            'coke_percent': coke_percent,
            'Sieving3mm_fuel': Sieving3mm_fuel,
            'CAO_limestone': cao_limestone,
            'charge': charge,
        }

        for param in params:
            params[param].index = params[param]['DATETIME']
            params[param].sort_index(ascending=False)
            params[param].drop(columns=['DATETIME'], inplace=True)

        return self.feature_engineering.aggregate(params)

    def __make_prediction(self, prepared_features):
        return float(self.model.predict(prepared_features))
