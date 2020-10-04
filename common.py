import os
import pandas as pd


class Common:
    @staticmethod
    def nowtime():
        """
        今の時間をstr形式で返すだけの関数
        :return:
        """
        from datetime import datetime
        return str(datetime.now().strftime('%Y%m%d%H%M%S'))

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
