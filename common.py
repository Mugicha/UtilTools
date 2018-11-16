import os

class Common:

    def range_check(self, _param, _min, _max):
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
            print('[folder_check] Exist -> ' + _path)
            return True
