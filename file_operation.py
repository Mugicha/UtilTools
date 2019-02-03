import pandas as pd
import os


class FileOperation:
    @staticmethod
    def get_file_list(_input_path: str, _is_recursive: bool = False, _can_return_abspath: bool = False):
        """
        指定したフォルダの中のファイルリストを作成し、配列として返す機能
        :param _input_path: ファイルリストを取得したいフォルダパス
        :param _is_recursive: 指定したフォルダの中にサブフォルダがある場合、そこも検索するか(default:検索しない)
        :param _can_return_abspath: 戻り値のファイルパスは絶対パスにするか(default: _input_pathからの相対パス)
        :return:
        """
        # Exit if input path is file path.
        if os.path.isfile(_input_path):
            print('[file_operation.py][get_file_list][Warn] Do not input file path.')
            exit(1)
        # 再帰的に検索
        if _is_recursive:
            fileList = []
            for root, dirs, files in os.walk(_input_path):
                for fr in files:
                    if os.path.isfile(os.path.join(root, fr)):
                        # return with absolute path.
                        if _can_return_abspath:
                            fileList.append(os.path.join(root, fr))
                        else:
                            local_root = os.path.split(_input_path)[1]
                            if os.name == 'nt':
                                ary = root.split('\\')
                            else:
                                ary = root.split('/')
                            if len(ary[ary.index(local_root)+1:-1]) == 0:
                                relative_path = './' + fr
                            else:
                                relative_path = './' + '/'.join(ary[ary.index(local_root)+1:-1]) + '/' + fr
                            fileList.append(relative_path)
            return fileList

        # 再帰的に検索しない
        else:
            files = os.listdir(_input_path)
            print(files)
            return [f for f in files if os.path.isfile(os.path.join(_input_path, f))]

    @staticmethod
    def detect_char_code(_path: str):
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
            return pd.read_csv(_path, parse_dates=[date_data_loc], date_parser=my_parser, encoding=self.detect_char_code(_path))
        try:
            return pd.read_csv(_path)
        except:
            print('Cannot import csv file. [' + _path + ']')
            exit(1)

    @staticmethod
    def df_to_csv(_df: pd.DataFrame, _path='./UtilTool.csv'):
        """
        dataframe を csvとして返す処理。
        :param _df:
        :param _path:
        :return:
        """
        _df.to_csv(_path, index=False)

    @staticmethod
    def excel_to_df(_input_file: str):
        """
        Excel の読み込み
        :param _input_file: 入力するExcelのファイルパス
        :return: Excelデータを格納したDataFrame(読み込めない場合はNone)
        """
        if not os.path.splitext(_input_file)[1] in ['.xlsx', '.xls']:
            print('[file_operation.py][excel_to_df] Input file must be xlsx or xls.')
            exit(1)
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

    @staticmethod
    def df_to_excel(_output_file: str, _output_df: pd.DataFrame, _sheet_name='exported'):
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
            print('[df_to_excel] Cannot export to excel.')

    @staticmethod
    def multiple_df_to_excel(_output_dir: str, _output_file: str, _output_df_list: list or dict):
        """
        DataframeをExcelの複数シートに保存するkに追う
        :param _output_dir: 保存先のディレクトリパス
        :param _output_file: 保存するExcelファイル名
        :param _output_df_list: Dataframeのリストもしくは辞書型変数。リストだとインデックス番号、辞書型の場合はkey名がシート名になる。
        :return:
        """
        from pandas import ExcelWriter
        try:
            with ExcelWriter(os.path.join(_output_dir, _output_dir)) as writer:
                if type(_output_df_list) == "list":
                    for idx, df in enumerate(_output_df_list):
                        df.to_excel(writer, 'sheets%s' % idx)
                        writer.save()
                elif type(_output_df_list) == 'dict':
                    for key in _output_df_list.keys():
                        _output_df_list[key].to_excel(writer, str(key))
                        writer.save()
        except:
            print('[multiple_df_to_excel] Cannot export to excel.')

    @staticmethod
    def txt_to_ary(_path: str):
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

    @staticmethod
    def ary_to_txt_1D(_list: list,
                      _output_file: str,
                      _mode: str,
                      _encoding: str = 'utf8'):
        """
        list型変数を要素ごとにテキストに書き出す。
        :param _list: 書き出したい配列LIST
        :param _output_file: 書き出し先
        :param _mode: 書き込みモード（w:書き出し, x:新規作成＆書き込み用に開く, a:末尾に追記）
        :param _encoding: 書き込み時の文字コード
        :return:
        """
        with open(_output_file, _mode, encoding=_encoding) as f:
            f.write('\n'.join(_list))
            f.write('\n')
        f.close()

    @staticmethod
    def ary_to_txt_2D(_list: list, _output_file: str, _mode: str, _encoding: str = 'utf8'):
        """

        :param _list: 書き出したい配列LIST（2次元)
        :param _output_file: 書き出し先ファイルパス
        :param _mode: 書き込みモード（w:書き出し, x:新規作成＆書き込み用に開く, a:末尾に追記）
        :param _encoding: 書き込み時の文字コード
        :return:
        """
        if len([len(v) for v in _list]) >= 3:  # return error if the size is more than 3d.
            print("[ary_to_txt_2D]: Can not export the list. Your list size is " + str([len(v) for v in _list]))
            return None
        with open(_output_file, _mode, encoding=_encoding) as f:
            for l in _list:
                f.write(','.join(l))
                f.write('\n')
        f.close()

    @staticmethod
    def dic_to_csv(_dic: dict, _output_file: str, _mode: str, _encoding: str = 'utf8'):
        """

        :param _dic: 書き出したい辞書型配列
        :param _output_file: 書き出し先ファイルパス
        :param _mode: 書き込みモード（w:書き出し, x:新規作成＆書き込み用に開く, a:末尾に追記）
        :param _encoding: 書き込み時の文字コード
        :return:
        """
        if str(type(_dic)) != "<class 'dict'>":
            print("[dic_to_txt]: Set _dic as dict type.")
            return None
        if _output_file.split('.')[-1] != 'csv':
            print("[dic_to_txt]: Set _output_file as csv file.")
            return None
        with open(_output_file, _mode, encoding=_encoding) as f:
            for key in _dic.keys():
                f.write(str(key) + ',' + str(_dic[key]))
                f.write('\n')
        f.close()
