import os

class Common:

    def range_check(self, _param, _min, _max):
        """
        変数が指定された
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
