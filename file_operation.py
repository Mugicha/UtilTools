import pandas as pd


class FileOperation():

    def csv_to_df(self, _path: str):
        """
        import csv and return the data as Dataframe.
        :param _path: csv path
        :return: DataFrame
        """
        return pd.read_csv(_path)
