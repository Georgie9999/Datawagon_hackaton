import sys
import numpy as np
import joblib
import pandas as pd
from lightgbm import Booster
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression


class Solution:

    def __init__(self, forecast_path=r'../data/forecast_example.csv'):
        self.forecast_example = pd.read_csv(forecast_path, sep=";")

    @staticmethod
    def add_master_data_mappings(fact_train_test: pd.DataFrame) -> pd.DataFrame:
        """

        :param fact_train_test:
        :return: преобразованые исходные данные -
        к таблице сделан merge дополнительной информации
        """
        client_mapping = pd.read_csv(r"../data/client_mapping.csv", sep=";")
        freight_mapping = pd.read_csv(r"../data/freight_mapping.csv", sep=";")
        station_mapping = pd.read_csv(r"../data/station_mapping.csv", sep=";")

        fact_train_test.period = fact_train_test.period.astype('category')
        fact_train_test.forecast_weight = list(map(lambda x: float(x.replace(",", ".")),
                                                   list(fact_train_test.forecast_weight)))

        fact_train_test = pd.merge(fact_train_test, client_mapping, how="left", on="client_sap_id")

        # Груз
        fact_train_test = pd.merge(fact_train_test, freight_mapping, how="left", on="freight_id")

        # Станции
        fact_train_test = pd.merge(
            fact_train_test,
            station_mapping.add_prefix("sender_"),
            how="left",
            on="sender_station_id",
        )
        fact_train_test = pd.merge(
            fact_train_test,
            station_mapping.add_prefix("recipient_"),
            how="left",
            on="recipient_station_id",
        )

        return fact_train_test

    def predict(self):
        """

        predict 3-х столбцов: recipient_station_id (направление),
        real_weight (перевозимый вес),
        real_wagon_count (число вагонов)
        """
        features_1 = ['rps', 'podrod', 'filial', 'client_sap_id',
            'freight_id', 'sender_station_id', 'sender_organisation_id',
            'holding_name', 'freight_group_name', 'sender_department_name', 'sender_railway_name']

        new_data = self.add_master_data_mappings(self.forecast_example)

        X = new_data.drop(["recipient_station_id", "forecast_weight", "forecast_wagon_count"], axis=1)
        model_recipient_station_id = Booster(model_file=r'../models/model_recipient_station_id.txt')
        model_real_weight = CatBoostRegressor().load_model(r"../models/weight_model.cbm")
        model_real_wagon_count = joblib.load(r"../models/reg_for_vagon_count.joblib")
        features_1 = ['period', 'rps', 'podrod', 'filial', 'client_sap_id', 'freight_id',
                      'sender_station_id', 'sender_organisation_id']
        X_f = X[features_1]
        print(X_f.info())
        recipient_station_id_pred = model_recipient_station_id.predict(X).astype(int)
        # real_wagon_pred = model_real_wagon_count.predict(new_data).astype(int)

        real_weight_pred = model_real_weight.predict(X_f)

        # real_wagon_pred = model_real_wagon_count.predict(real_weight_pred).astype(int)

        real_weight_pred = list(map(lambda x: str(x).replace(".", ","), real_weight_pred))

        # записываем результат в таблицу

        new_data.forecast_weight = list(real_weight_pred)
        # new_data.forecast_wagon_count = list(real_wagon_pred)
        new_data.recipient_station_id = list(recipient_station_id_pred)
        new_data.to_csv("result.csv", sep=";", decimal=',', encoding='windows-1251')


example = Solution()
example.predict()
