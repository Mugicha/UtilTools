import os
import pandas as pd


class FileOperation:
    @staticmethod
    def get_file_list(_input_path: str, _is_recursive: bool = False, _can_return_abspath: bool = False):
        """
        指定したフォルダの中のファイルリストを作成し、配列として返す機能
        :param _input_path: ファイルリストを取得したいフォルダパス
        :param _is_recursive: 指定したフォルダの中にサブフォルダがある場合、そこも検索するか(default:検索しない)
        :param _can_return_abspath: 戻り値のファイルパスは絶対パスにするか(default: _input_pathからの相対パス)
        :return: list型オブジェクト.
        """
        # Exit if input path is file path.
        if os.path.isfile(_input_path):
            assert '[file_operation.py][get_file_list][Warn] Do not input file path.'
            return []
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

    def csv_to_df(self, _path: str, date_convert=False, date_format='YYYY-mm-dd', date_data_loc: int = 0, header=None):
        """
        import csv and return the data as DataFrame.
        :param _path: csv path
        :param date_convert: convert flag if the data contain date format data.
        :param date_format: date data format e.g. YYYY-mm-dd
        :param date_data_loc: date data column location in csv file.
        :param header: row num to use column name (default: None which means no header).
        :return: DataFrame
        """
        ext = os.path.splitext(_path)[1]
        if not ext in ['.csv', '.tsv']:
            assert '[csv_to_df] Extension must be csv or tsv.'
            return None
        sep = ',' if ext == '.csv' else '\t'
        if date_convert:
            my_parser = lambda date: pd.datetime.strptime(date, date_format)
            return pd.read_csv(_path, parse_dates=[date_data_loc], sep=sep,
                               date_parser=my_parser, encoding=self.detect_char_code(_path), header=header)
        return pd.read_csv(_path, header=header, sep=sep)

    @staticmethod
    def df_to_csv(_df: pd.DataFrame, _output_dir: str = './', _output_file: str = 'UtilTool.csv', _encode: str = 'utf8'):
        """
        dataframe を csvとして返す処理。
        :param _df:
        :param _output_dir:
        :param _output_file:
        :param _encode: char encode type (default: utf-8)
        :return:
        """
        if os.path.splitext(_output_file)[1] != '.csv':
            assert '[df_to_csv] Extension must be csv.'
            return None
        _df.to_csv(os.path.join(_output_dir, _output_file), index=False, encoding=_encode)

    @staticmethod
    def excel_to_df(_input_path: str, _header: int = 0):
        """
        Excel の読み込み
        :param _input_path: 入力するExcelのファイルパス
        :param _header: ヘッダーの行（0スタート)
        :return: Excelデータを格納したDataFrame(読み込めない場合はNone)
        """
        if not os.path.splitext(_input_path)[1] in ['.xlsx', '.xls']:
            assert '[file_operation.py][excel_to_df] Input file must be xlsx or xls.'
        entire_row_data = pd.ExcelFile(_input_path)
        sheet_names = entire_row_data.sheet_names
        if len(sheet_names) == 1:  # シートが1つの時
            row_data = entire_row_data.parse(sheet_names[0], header=_header)
        elif len(sheet_names) > 1:  # 複数のシートが存在する場合
            print('シートが複数存在します。読み込むシートの番号を下記から選択してください。')
            for sht_num in range(len(sheet_names)):
                print(str(sht_num) + ' : ' + sheet_names[sht_num])
            input_sht_num = input('>>> ')
            row_data = entire_row_data.parse(sheet_names[int(input_sht_num)], header=_header)
        else:
            return None
        return row_data

    @staticmethod
    def df_to_excel(_df: pd.DataFrame,
                    _output_dir: str = './',
                    _output_file: str = 'output.xlsx',
                    _sheet_name='exported',
                    _encoding: str = 'utf8',
                    _index: bool = False,
                    ):
        """
        Excelの書き出し
        :param _output_dir:
        :param _output_file: 出力するExcelのファイル名
        :param _df: 出力するデータ(DataFrame型)
        :param _sheet_name: exportしたexcelのシート名
        :param _encoding: エクスポート時の文字コード（default: utf-8)
        :param _index: excel出力時にインデックスもexportするかどうか。
        :return: always None
        """
        if os.path.splitext(_output_file)[1] != '.xlsx':
            assert '[df_to_excel] Extension must be xlsx.'
            return None
        _df.to_excel(os.path.join(_output_dir, _output_file),
                     sheet_name=_sheet_name,
                     index=_index,
                     encoding=_encoding,
                     )

    @staticmethod
    def multiple_df_to_excel(_output_dir: str,
                             _output_file: str,
                             _output_df_list: list or dict,
                             ):
        """
        DataframeをExcelの複数シートに保存する機能。
        :param _output_dir: 保存先のディレクトリパス
        :param _output_file: 保存するExcelファイル名
        :param _output_df_list: Dataframeのリストもしくは辞書型変数。リストだとインデックス番号、辞書型の場合はkey名がシート名になる。
        :return:
        """
        from pandas import ExcelWriter
        with ExcelWriter(os.path.join(_output_dir, _output_file), engine='openpyxl') as writer:
            if "list" in str(type(_output_df_list)):
                for idx, df in enumerate(_output_df_list):
                    df.to_excel(writer, 'sheets%s' % idx)
                    writer.save()
            elif 'dict' in str(type(_output_df_list)):
                for key in _output_df_list.keys():
                    _output_df_list[key].to_excel(writer, str(key), index=False)
                    writer.save()

    @staticmethod
    def txt_to_ary(_path: str,
                   _encoding: str = 'utf8',
                   ):
        """
        対象のファイルを1行ごとに配列に格納してreturnする。
        改行は配列に入れる際に削除される。
        :param _path: 配列に格納したファイルのパス
        :param _encoding: file encoding
        :return: 1行1要素として格納された配列
        """
        _list = []
        with open(_path, encoding=_encoding) as f:
            for line in f:
                _list.append(line.replace('\n', ''))
        return _list  # type: list

    @staticmethod
    def ary_to_txt_1D(_list: list,
                      _output_file: str,
                      _mode: str,
                      _encoding: str = 'utf8',
                      ):
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
    def ary_to_txt_2D(_list: list,
                      _output_file: str,
                      _mode: str,
                      _encoding: str = 'utf8',
                      ):
        """

        :param _list: 書き出したい配列LIST（2次元)
        :param _output_file: 書き出し先ファイルパス
        :param _mode: 書き込みモード（w:書き出し, x:新規作成＆書き込み用に開く, a:末尾に追記）
        :param _encoding: 書き込み時の文字コード
        :return:
        """
        if len([len(v) for v in _list]) >= 3:  # return error if the size is more than 3d.
            assert f"[ary_to_txt_2D]: Can not export the list. Your list size is {[len(v) for v in _list]}"
            return None
        with open(_output_file, _mode, encoding=_encoding) as f:
            for l in _list:
                f.write(','.join(l))
                f.write('\n')
        f.close()

    @staticmethod
    def dic_to_csv(_dic: dict,
                   _output_file: str,
                   _mode: str,
                   _encoding: str = 'utf8',
                   ):
        """
        辞書型配列のkeyとvalueを2列のcsvにして書き出す機能。
        :param _dic: 書き出したい辞書型配列
        :param _output_file: 書き出し先ファイルパス
        :param _mode: 書き込みモード（w:書き出し, x:新規作成＆書き込み用に開く, a:末尾に追記）
        :param _encoding: 書き込み時の文字コード
        :return:
        """
        if str(type(_dic)) != "<class 'dict'>":
            assert "[dic_to_txt]: Set _dic as dict type."
            return None
        if _output_file.split('.')[-1] != 'csv':
            assert "[dic_to_txt]: Set _output_file as csv file."
            return None
        with open(_output_file, _mode, encoding=_encoding) as f:
            for key in _dic.keys():
                f.write(str(key) + ',' + str(_dic[key]))
                f.write('\n')
        f.close()

    def excel_to_dic(self, _input_excel_file_path: str):
        """
        excelの1列目をkey, 2列目をValueとしてdictionaryに変換する機能
        :param _input_excel_file_path:
        :return:
        """
        if os.path.splitext(_input_excel_file_path)[1] != '.xlsx':
            assert '[df_to_excel] Extension must be xlsx.'
            return None
        df = self.excel_to_df(_input_excel_file_path)
        dic = {}
        for idx, row in df.iterrows():
            dic[row.iloc[0]] = row.iloc[1]
        return dic
