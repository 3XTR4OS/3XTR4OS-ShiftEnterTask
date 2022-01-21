import pandas as pd
import numpy as np


class FeatureEngineeringCAO:

    def aggregate(self, input_data):

        params = self.__get_params(input_data)
        cleaned_params = self.__fill_NA_values(params)
        features = self.__create_features(cleaned_params)

        return features

    def __get_params(self, input_data):
        frequent_df = input_data['limestone_consumption']

        frequent_params = ['concentrate_limestone_consumption',
                           'charge_consumption', 'coke_percent']

        for param in frequent_params:
            frequent_df = pd.merge_asof(right=frequent_df.sort_index(),
                                        left=input_data[param].sort_index(),
                                        left_index=True,
                                        right_index=True,
                                        tolerance=pd.Timedelta('5m'),
                                        direction='nearest')

        frequent_df.dropna(inplace=True)

        params_df = pd.DataFrame()
        params_df['Расход кокса'] = (frequent_df['charge_consumption'] *
                                     frequent_df['coke_percent'] * 0.01).resample('60T').mean()

        params_df['Расход извести + концентрата'] = (
            frequent_df['concentrate_limestone_consumption'].resample('60T').mean())

        # ближайшие к часу
        params_df['Расход извести'] = (
            frequent_df[frequent_df.index.minute == 0]['limestone_consumption'])

        params_df['Расход концентрата'] = (
            params_df['Расход извести + концентрата'] -
            params_df['Расход извести'])

        params_df['Расход шихты '] = (
            params_df['Расход извести + концентрата'] -
            params_df['Расход кокса'])

        params_df['Concentrate'] = (
            params_df['Расход концентрата'] /
            params_df['Расход шихты '])

        columns_to_delete = ['Расход кокса', 'Расход извести + концентрата',
                             'Расход извести', 'Расход концентрата',
                             'Расход шихты ']

        params_df.drop(columns=columns_to_delete, inplace=True)

        charge_df = input_data['charge']

        params_df = pd.merge_asof(
            left=params_df.sort_index(),
            right=charge_df.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta('30m'),
            direction='nearest',
        )
        params_df.dropna(inplace=True)

        sieveing_df = input_data['Sieving3mm_fuel']
        cao_limestone_df = input_data['CAO_limestone']

        date_time_list = (list(sieveing_df.index) + list(cao_limestone_df.index)
                          + list(params_df.index))

        time_indecies = [
            x for x in pd.date_range(
                min(date_time_list) - pd.Timedelta('1h'),
                max(date_time_list) + pd.Timedelta('1h'),
                freq='H')]

        temp_df = pd.DataFrame(data=np.zeros(len(time_indecies)),
                               index=time_indecies)

        temp_df = temp_df.merge(sieveing_df,
                                left_index=True,
                                right_index=True,
                                how='outer')

        temp_df = temp_df.merge(cao_limestone_df,
                                left_index=True,
                                right_index=True,
                                how='outer')

        for col in temp_df.columns:
            temp_df[col] = temp_df[col].replace({0: None})
            temp_df[col] = temp_df[col].astype(float).interpolate()

        temp_df = temp_df.drop(columns=[0])

        params_df = pd.merge_asof(
            left=params_df,
            right=temp_df,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta('30m'),
            direction='nearest',)

        params_df['DATETIME'] = params_df.index
        params_df.reset_index(drop=True, inplace=True)

        return params_df

    def __fill_NA_values(self, df):

        df = df.fillna(10.0)
        df = df.reset_index(drop=True)

        return df

    def __create_features(self, params):
        columns_rolling_4 = ['CAO_charge', 'Concentrate', 'OSN_charge',
                             'TIO2_charge', 'Sieving3mm_fuel']
        columns_rolling_3 = []
        columns_rolling_2 = ['CAO_charge', 'CAO_limestone']
        rolls = [columns_rolling_2, columns_rolling_3, columns_rolling_4]
        params_with_rolling = params.copy()

        for number, rol in enumerate(rolls):
            for col in rol:
                try:
                    params_with_rolling[[col + f"_rolling_{number + 2}"]] = (
                        params_with_rolling[[col]].rolling(number + 2).mean())

                except (RuntimeError, TypeError, NameError):
                    pass

        params_with_rolling = self.__fill_NA_values(params_with_rolling)

        features = pd.DataFrame()
        features['CAO_charge_rolling_2'] = (
            params_with_rolling['CAO_charge_rolling_2'])

        features['log(Concentrate_rolling_4)/CAO_charge_rolling_4'] = (
            (np.log(params_with_rolling['Concentrate_rolling_4']))
                .divide(params_with_rolling['CAO_charge_rolling_4']))

        features['CAO_charge_rolling_2**3*OSN_charge_rolling_4'] = (
            (params_with_rolling['CAO_charge_rolling_2'] ** 3)
                .multiply(params_with_rolling['OSN_charge_rolling_4']))

        features['CAO_charge/TIO2_charge_rolling_4'] = (
            (params_with_rolling['CAO_charge'])
                .divide(params_with_rolling['TIO2_charge_rolling_4']))

        features['CAO_charge**3*log(Sieving3mm_fuel_rolling_4)'] = (
            (params_with_rolling['CAO_charge'] ** 3)
                .multiply(
                np.log(params_with_rolling['Sieving3mm_fuel_rolling_4'])))

        features['sqrt(CAO_limestone_rolling_2)*CAO_charge**3'] = (
            (np.sqrt(params_with_rolling['CAO_limestone_rolling_2']))
                .multiply((params_with_rolling['CAO_charge'] ** 3)))

        return features
