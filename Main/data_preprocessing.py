import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
import warnings
from config import Config
import lightgbm as lgb


class DataReader:
    """
    Class for reading and preprocessing data
    """
    _is_cleaned = False
    __CODER = None

    @staticmethod
    def __transform_benz(data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforming dataframe column Benz using mapping and dict
        :param data: dataframe to transform
        :return: transformed dataframe:
        """
        dct = {"Дизель": 'dis', 'Бензин': 'benz', 'Газ / Бензин': 'gb', "Газ": 'gas', 'Гібрид': 'gibr',
               'Електро': 'elctr',
               'Газ пропан-бутан': 'gpb', 'Газ метан': 'gb'}
        return data.replace({'Benz': dct})

    @staticmethod
    def __transform_trans(data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforming dataframe column Transmission using mapping and dict
        :param data: dataframe to transform
        :return: transformed dataframe
        """
        dct = {'Автомат': 'auto', 'Ручна/Механіка': 'mech', 'Варіатор': 'var', 'Типтронік': 'tip', 'Робот': 'rob'}
        return data.replace({'Transmission': dct})

    @staticmethod
    def create_lgb_dataset(data:pd.DataFrame) -> lgb.Dataset:
        """
        Creating lightgbm dataset with label Price_USD
        :param data: dataframe
        :return: lightgbm dataset
        """
        return lgb.Dataset(data=data.drop('Price_USD', axis=1), label=data.Price_USD)
    def read_cars_data(self, conf:Config) -> pd.DataFrame:
        """
        reading and cleaning used cars data
        :param conf: file path
        :return: cleaned pandas Dataframe
        """
        df = pd.read_csv(conf.main_data)

        # droping local brands
        ukr_indexes = df[df.Brand.apply(lambda x: self._hasukr(x))].index
        df = df.drop(ukr_indexes)

        # dropping brands with small amount of cars
        df = df.drop(df[df.Brand.isin(df.Brand.value_counts()[df.Brand.value_counts() < 200].index)].index)
        # dropping small cities
        df = df[df.City.isin(df.City.value_counts()[df.City.value_counts() > 200].index)]
        df = df.reset_index()
        df = df.reset_index()

        # creating column for special cars
        ukr_models = df[df.Model.apply(lambda x: self._hasukr(x))].Model.str.rstrip('.')
        ukr_models.drop(ukr_models[(ukr_models.str.contains('СС')) | (ukr_models.str.contains('GХ2')) | (
            ukr_models.str.contains('С'))].index, inplace=True)
        ukr_models = ukr_models.str.rsplit(' ', n=1).str[0]
        df['Special'] = False
        df['Special'][ukr_models.index] = True
        df['Model'][ukr_models.index] = ukr_models

        # transforming features in ukrainian to english
        df = DataReader.__transform_benz(df)
        df = DataReader.__transform_trans(df)

        # cleaning  data
        df.drop(df.Run[df.Run > 999].index, inplace=True)
        df.drop(df[df.Year < 1975].index, inplace=True)
        df.drop(df.L[df.L == 'Не'].index, inplace=True)

        df.drop(df[df.iloc[:, :-1].duplicated()].index, inplace=True)
        self._is_cleaned = True

        return df

    def data_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scaling and encoding dataframe
        :param df: cleaned dataframe
        :return: binary encoded dataframe
        """
        if not self._is_cleaned:
            warnings.warn("Your data is not cleaned. Read data through read_cars_data() ")

        # Scaling Price_USD feature for normal dist
        df['Price_USD'] = np.log1p(df['Price_USD'])

        # removing unknown litres ih L
        df.L[df.L == 'Unknown'] = '-100'
        df.L = df.L.astype('float64')

        # encoding text features
        cat_cols = [
            'Brand',
            'City',
            'Model',
            'Benz',
            'Transmission'
        ]

        encoder = BinaryEncoder(cols=cat_cols, return_df=True, verbose=False)
        self.__CODER = encoder
        binary_encoded_df = encoder.fit_transform(df)
        binary_encoded_df.drop(['level_0', 'index'], axis=1, inplace=True)
        return binary_encoded_df

    def get_encoder(self) -> BinaryEncoder:
        return self.__CODER

    def _hasukr(self, s) -> bool:
        """
        Check if ukrainian in string
        :param s: string
        """
        lower = set('абвгґдеєжзиіїйклмнопрстуфхцчшщьюя')
        return lower.intersection(s.lower()) != set()
