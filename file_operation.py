import pandas as pd


class FileOperation():

    def detect_char_code(self, _path: str):
        """
        指定したファイルの文字コードを判別して返す機能
        :param _path: 文字コードを判別したいファイル
        :return: 文字コード
        """
        import chardet
        with open(_path, mode='rb') as f:
            binary = f.readline()
            result = chardet.detect(binary)
            return result['encoding']

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
            #try:
            return pd.read_csv(_path, parse_dates=[date_data_loc], date_parser=my_parser, encoding=self.detect_char_code(_path))
            #except:
            #    print('Cannot import csv file or convert time format.')
            #    exit(1)
        try:
            return pd.read_csv(_path)
        except:
            print('Cannot import csv file. [' + _path + ']')
            exit(1)

    def df_to_csv(self, _df: pd.DataFrame, _path='./UtilTool.csv'):
        _df.to_csv(_path, index=False)

    def excel_to_df(self, _input_file: str):
        """
        Excel の読み込み
        :param _input_file: 入力するExcelのファイルパス
        :return: Excelデータを格納したDataFrame(読み込めない場合はNone)
        """
        try:
            entire_row_data = pd.ExcelFile(_input_file, encoding='utf8')
        except:
            return None
        sheet_names = entire_row_data.sheet_names
        if len(sheet_names) == 1:  # シートが1つの時
            row_data = entire_row_data.parse(sheet_names[0])
        elif len(sheet_names) > 1:  # 複数のシートが存在する場合
            print('シートが複数存在します。読み込むシートの番号を下記から選択してください。')
            for sht_num in range(len(sheet_names)):
                print(str(sht_num) + ' : ' + sheet_names[sht_num])
            input_sht_num = input('>>> ')
            row_data = entire_row_data.parse(sheet_names[int(input_sht_num)])
        else:
            return None
        return row_data

    def df_to_excel(self, _output_file: str, _output_df: pd.DataFrame, _sheet_name='exported'):
        """
        Excelの書き出し
        :param _output_file: 出力するExcelのファイルパス
        :param _output_df: 出力するデータ(DataFrame型)
        :param _sheet_name: exportしたexcelのシート名
        :return: 終了コード（0:正常, 1:異常）
        """
        try:
            _output_df.to_excel(_output_file, sheet_name=_sheet_name, index=False)
        except:
            print('Cannot export to excel.')
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

    def ary_to_txt(self, _list: list, _path: str, _mode: str):
        """
        list型変数を要素ごとにテキストに書き出す。
        :param _list: 書き出したい配列LIST
        :param _path: 書き出し先
        :param _mode: 書き込みモード（w:書き出し, x:新規作成＆書き込み用に開く, a:末尾に追記）
        :return:
        """
        with open(_path, _mode) as f:
            f.write('\n'.join(_list))
            f.write('\n')
        f.close()
