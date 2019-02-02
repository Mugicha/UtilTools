import os
import pandas as pd


class Common:
    def extract_all_combination(self, _df: pd.DataFrame, _combination_pare_column: list):
        """
        データフレームの指定した列内に出現する値の総組合せを配列で返す機能。
        まだ試作段階。
        :param _df:
        :param _combination_pare_column:
        :return:
        """
        import itertools
        num_of_param = len(_combination_pare_column)
        return_ary = []
        if num_of_param == 2:
            p1 = self.remove_duplication(_df[_combination_pare_column[0]].values)
            p2 = self.remove_duplication(_df[_combination_pare_column[1]].values)
            for _1, _2 in itertools.product(p1, p2):
                return_ary.append([_1, _2])
        elif num_of_param == 3:
            p1 = self.remove_duplication(_df[_combination_pare_column[0]].values)
            p2 = self.remove_duplication(_df[_combination_pare_column[1]].values)
            p3 = self.remove_duplication(_df[_combination_pare_column[2]].values)
            for _1, _2, _3 in itertools.product(p1, p2, p3):
                return_ary.append([_1, _2, _3])
        elif num_of_param == 4:
            p1 = self.remove_duplication(_df[_combination_pare_column[0]].values)
            p2 = self.remove_duplication(_df[_combination_pare_column[1]].values)
            p3 = self.remove_duplication(_df[_combination_pare_column[2]].values)
            p4 = self.remove_duplication(_df[_combination_pare_column[3]].values)
            for _1, _2, _3, _4 in itertools.product(p1, p2, p3, p4):
                return_ary.append([_1, _2, _3, _4])
        elif num_of_param == 5:
            p1 = self.remove_duplication(_df[_combination_pare_column[0]].values)
            p2 = self.remove_duplication(_df[_combination_pare_column[1]].values)
            p3 = self.remove_duplication(_df[_combination_pare_column[2]].values)
            p4 = self.remove_duplication(_df[_combination_pare_column[3]].values)
            p5 = self.remove_duplication(_df[_combination_pare_column[4]].values)
            for _1, _2, _3, _4, _5 in itertools.product(p1, p2, p3, p4, p5):
                return_ary.append([_1, _2, _3, _4, _5])
        else:
            print('The number of params you specified is too many. Reduce less than 5.')
            return None
        return return_ary  # type: list

    @staticmethod
    def range_check(_param, _min, _max):
        """
        変数が指定された範囲内にいるか確認する機能
        :param _param:
        :param _min:
        :param _max:
        :return:
        """
        if _min <= _param <= _max:
            return True
        else:
            return False

    @staticmethod
    def folder_check(_path):
        """
        フォルダの存在確認をしてくれる機能
        :param _path: 確認したいフォルダのパス
        :return:
        """
        if not os.path.isdir(_path):
            try:
                os.mkdir(_path)
                print('[common.py][folder_check] Create -> ' + _path)
                return True
            except:
                print('[common.py][Warn] Cannot create folder. ' + _path)
                return False
        else:
            return True

    @staticmethod
    def remove_duplication(_list: list):
        """
        リストの重複を削除する機能
        :param _list: 重複を削除したい配列
        :return: 重複を削除した配列
        """
        seen = set()
        seen_add = seen.add
        return [x for x in _list if x not in seen and not seen_add(x)]

    @staticmethod
    def file_exist_check(_file_path):
        """
        ファイルが存在するかどうか確認し、存在するならファイル名の末尾にカウンター文字を付与して返す機能
        e.g.
        hoge.txt を_file_pathに入力し、すでにhoge.txtが存在していれば hoge_(1).txtを返す。
        :param _file_path: 存在確認したいファイルパス
        :return: 保存して大丈夫なファイルパス
        """
        if os.path.exists(_file_path):
            base = os.path.dirname(_file_path)
            name = os.path.basename(_file_path).split('.')
            counter = 1
            cwd = os.getcwd()
            while os.path.exists(os.path.join(cwd, base, ''.join(name[0:-1]) + '_(' + str(counter) + ').' + str(name[-1]))):
                counter += 1
            return os.path.join(cwd, base, ''.join(name[0:-1]) + '_(' + str(counter) + ').' + str(name[-1]))
        else:
            return _file_path
