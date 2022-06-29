import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from logger import logger
from datetime import date, datetime


class CleanDataFrame:

    @staticmethod
    def get_numerical_columns(df: pd.DataFrame) -> list:
        try:
            numerical_columns = df.select_dtypes(
                include='number').columns.tolist()
            logger.info('successfully got numerical columns')
            return numerical_columns
        except Exception as e:
            logger.error(e)
            return []

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        try:
            categorical_columns = df.select_dtypes(
                include=['object']).columns.tolist()
            logger.info('successfully got catagorical columns')
            return categorical_columns
        except Exception as e:
            logger.error(e)
            return []

    def fix_datatypes(self, df: pd.DataFrame, column: str = None, to_type: type = None) -> pd.DataFrame:
        """
        Takes in the sales dataframe an casts columns to proper data type
        """
        try:
            datetime_columns = ['Date']
            string_columns = ['PromoInterval',
                              'StoreType', 'Assortment', 'StateHoliday']
            int_columns = ['CompetitionOpenSinceYear',
                           'CompetitionOpenSinceMonth', 'Promo2SinceWeek', 'Promo2SinceYear']
            df_columns = df.columns
            for col in string_columns:
                if col in df_columns:
                    df[col] = df[col].astype("string")
                logger.info(f'successfully changed {col} column to string')
            for col in datetime_columns:
                if col in df_columns:
                    df[col] = pd.to_datetime(df[col])
                logger.info(f'successfully changed {col} column to datetime')
            for col in int_columns:
                if col in df_columns:
                    df[col] = df[col].astype("int64")
                logger.info(f'successfully changed {col} column to integer')
            if column and to_type:
                df[column] = df[column].astype(to_type)
                logger.info(f'successfully changed {col} column to {to_type}')
            logger.info('successfully finished fixing datatype')
        except Exception as e:
            logger.error(e)
        return df

    def percent_missing(self, df):
        """
        Print out the percentage of missing entries in a dataframe
        """
        try:
            # Calculate total number of cells in dataframe
            totalCells = np.product(df.shape)

            # Count number of missing values per column
            missingCount = df.isnull().sum()

            # Calculate total number of missing values
            totalMissing = missingCount.sum()

            # Calculate percentage of missing values
            logger.info(
                f"The dataset contains {round(((totalMissing/totalCells) * 100), 2)} % missing values.")
            return round(((totalMissing/totalCells) * 100), 2)
        except Exception as e:
            logger.error(e)

    def get_mct(self, series: pd.Series, measure: str):
        """
        get mean, median or mode depending on measure
        """
        try:
            measure = measure.lower()
            if measure == "mean":
                return series.mean()
            elif measure == "median":
                return series.median()
            elif measure == "mode":
                return series.mode()[0]
        except Exception as e:
            logger.error(e)

    def replace_missing(self, df: pd.DataFrame, columns: str, method: str) -> pd.DataFrame:
        try:
            for column in columns:
                nulls = df[column].isnull()
                indecies = [i for i, v in zip(nulls.index, nulls.values) if v]
                replace_with = self.get_mct(df[column], method)
                df.loc[indecies, column] = replace_with
                logger.info(
                    f'successfully replaced missing values of {column} column with {method} method')
        except Exception as e:
            logger.error(e)
        return df

    def fix_missing_values(self, df: pd.DataFrame, columns: list, value) -> pd.DataFrame:
        try:
            for column in columns:
                df[column] = df[column].fillna(value)
                logger.info(
                    f'successfully fixed missing values of {column} column with {value}')
        except Exception as e:
            logger.error(e)
        return df

    def replace_value(self, df: pd.DataFrame, column, val_before, val_to):
        try:
            df.loc[df[column] == val_before, column] = val_to
            logger.info(
                f'successfully replaced value of {column} column with {val_to}')
        except Exception as e:
            logger.error(e)
        return df

    def remove_null_row(self, df: pd.DataFrame, columns: str) -> pd.DataFrame:
        try:
            for column in columns:
                df = df[~ df[column].isna()]
                logger.info(
                    f'successfully removed empty rows with null {column} column')
        except Exception as e:
            logger.error(e)
        return df

    def normal_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            scaller = StandardScaler()
            scalled = pd.DataFrame(scaller.fit_transform(
                df[self.get_numerical_columns(df)]))
            scalled.columns = self.get_numerical_columns(df)
            logger.info(f'successfully scalled the dataframe')
            return scalled
        except Exception as e:
            logger.error(e)
            return df

    def find_before_holiday_dates(self, df):
        try:
            today = date.today()
            christmass = datetime.strptime(
                '2015-12-25', '%Y-%m-%d').date().replace(year=today.year)
            easter = datetime.strptime(
                '2015-04-09', '%Y-%m-%d').date().replace(year=today.year)
            store_value = x['Store'].values
            x = x['Date'].values
            converted_dates = []
            response_array = []
            for index, value in enumerate(x):
                exp = df[df['Store'] == store_value[index]]
                exp = exp[exp["StateHoliday"] != '0'].sort_values(by="Date")
                the_dates = exp['Date'].unique()
                converted_dates = []
                for day in dates:
                    converted_dates.append(datetime.strptime(
                        str(day).split('T')[0], '%Y-%m-%d'))
                value = datetime.strptime(str(value).split('T')[0], '%Y-%m-%d')

                if value in converted_dates:
                    response_array.append(0)
                else:
                    above_dates = []
                    for d in converted_dates:
                        if d > value:
                            above_dates.append(d)
                    if len(above_dates) != 0:
                        the_date = above_dates[0]
                        response_array.append(abs((the_date - value).days))
                    else:
                        value = value.date().replace(year=today.year)
                        christmass_diff = (christmass - value).days
                        easter_diff = (easter - value).days
                        if christmass_diff < 0:
                            christmass_diff = 365 + christmass_diff
                        if easter_diff < 0:
                            easter_diff = 365 + easter_diff
                        if(easter_diff < christmass_diff):
                            diff = easter_diff
                        elif(easter_diff >= christmass_diff):
                            diff = christmass_diff
                        response_array.append(diff)
        except Exception as e:
            logger.error(e)
        return response_array

    def scale_column(self, df, column: str, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
        """
        try:
            std_column_df = pd.DataFrame(df[column])
            std_column_values = std_column_df.values
            minmax_scaler = MinMaxScaler(feature_range=range_tup)
            normalized_data = minmax_scaler.fit_transform(std_column_values)
            df[column] = normalized_data
            return df
        except Exception as e:
            logger.error(e)

    def scale_columns(self, df, columns: list, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        try:
            for col in columns:
                df = self.scale_column(df, col, range_tup)

            return df
        except Exception as e:
            logger.error(e)

    def change_datatypes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        A simple function which changes the data types of the dataframe and returns it
        """
        try:
            data_types = dataframe.dtypes
            changes = ['float64', 'int64']
            for col in data_types.index:
                if(data_types[col] in changes):
                    if(data_types[col] == 'float64'):
                        dataframe[col] = pd.to_numeric(
                            dataframe[col], downcast='float')
                    elif(data_types[col] == 'int64'):
                        dataframe[col] = pd.to_numeric(
                            dataframe[col], downcast='unsigned')
        except Exception as e:
            logger.error(e)

        return dataframe

    def minmax_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            scaller = MinMaxScaler()
            scalled = pd.DataFrame(
                scaller.fit_transform(
                    df[self.get_numerical_columns(df)]),
                columns=self.get_numerical_columns(df)
            )

            return scalled
        except Exception as e:
            logger.error(e)
            return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            normalizer = Normalizer()
            normalized = pd.DataFrame(
                normalizer.fit_transform(
                    df[self.get_numerical_columns(df)]),
                columns=self.get_numerical_columns(df)
            )
            logger.info(f'successfully normalized dataframe')
            return normalized
        except Exception as e:
            logger.error(e)
            return df

    def get_difference(self, dataset, interval=1):
        """
         A function to get the difference of scaled data
        """
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    def drop_duplicates(self, df: pd.DataFrame, subset=None) -> pd.DataFrame:
        """
        This checkes if there are any duplicated entries for a user
        And remove the duplicated rows
        """
        try:
            if subset != None:
                df = df.drop_duplicates(subset='')
            else:
                df = df.drop_duplicates()
            logger.info(f'successfully droped duplicates')
        except Exception as e:
            logger.error(e)
        return df

    def drop_columns(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Drops columns that are not essesntial for modeling
        """
        try:
            if len(columns) != 0:
                df.drop(columns=columns, inplace=True)
            logger.info(f'successfully dropped columns')
        except Exception as e:
            logger.error(e)
        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This runs a series of cleaner methods on the df passed to it. 
        """
        try:
            df = self.drop_duplicates(df)
            # df = self.drop_columns(df)
            df.reset_index(drop=True, inplace=True)
            logger.info(f'successfully finished cleaning pipeline')
        except Exception as e:
            logger.error(e)
        return df
