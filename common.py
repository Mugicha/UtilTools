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

    def range_check(self, _param, _min, _max):
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

    def folder_check(self, _path):
        """
        フォルダの存在確認をしてくれる機能
        :param _path: 確認したいフォルダのパス
        :return:
        """
        if not os.path.isdir(_path):
            try:
                os.mkdir(_path)
                print('[folder_check] Create -> ' + _path)
                return True
            except:
                print('[common.py][Warn] Cannot create folder. ' + _path)
                return False
        else:
            # print('[folder_check] Exist -> ' + _path)
            return True

    def remove_duplication(self, _list: list):
        """
        リストの重複を削除する機能
        :param _list: 重複を削除したい配列
        :return: 重複を削除した配列
        """
        seen = set()
        seen_add = seen.add
        return [x for x in _list if x not in seen and not seen_add(x)]
