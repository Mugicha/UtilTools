import pandas as pd


class FileOperation():

    def csv_to_df(self, _path: str, date_convert=False, date_format='YYYY-mm-dd', date_data_loc=0):
        """
        import csv and return the data as DataFrame.
        :param _path: csv path
        :param date_convert: convert flag if the data contain date format data.
        :param date_format: date data format e.g. YYYY-mm-dd
        :param date_data_loc: date data column location in csv file.
        :return: DataFrame
        """
        if date_convert:
            my_parser = lambda date: pd.datetime.strptime(date, date_format)
            try:
                return pd.read_csv(_path, parse_dates=[date_data_loc], date_parser=my_parser)
            except:
                print('Cannot import csv file or convert time format.')
                exit(1)
        try:
            return pd.read_csv(_path)
        except:
            print('Cannot import csv file. [' + _path + ']')
            exit(1)

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
        return _list  # type: list
