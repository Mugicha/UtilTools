import pandas as pd


class FileOperation():

    def csv_to_df(self, _path: str):
        """
        import csv and return the data as Dataframe.
        :param _path: csv path
        :return: DataFrame
        """
        return pd.read_csv(_path)

    def txt_to_ary(self, _path: str):
        """
        対象のファイルを1行ごとに配列に格納してreturnする。
        改行は配列に入れる際に削除される。
        :param _path: 配列に格納したファイルのパス
        :return: 1行1要素として格納された配列
        """
        _list = []
        with open(_path) as f:
            for line in f:
                _list.append(line.replace('\n', ''))
        return _list
