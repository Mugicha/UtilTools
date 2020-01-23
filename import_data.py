import pathlib
import glob
import MeCab


class ImportData:
    def __init__(self):
        pass

    def import_livedoor(self, _path: str):
        """
        Livedoor newsデータを読み込む処理.
        :param _path:
        :return: [['今日', 'は', '熱い'], ['今日', 'は', '眠い'], ['今日', 'は', '寒い'], ]
        """
        pth = pathlib.Path(_path)
        label = []
        m = MeCab.Tagger('-Owakati')
        for p in pth.glob('**/*.txt'):
            if p.name in ['CHANGES.txt', 'README.txt', 'LICENSE.txt']:
                continue
            with open(p, 'r', encoding='utf-8-sig') as f:
                buff = f.read()
                label.extend(m.parse(buff).split(' ')[:-1])
        return label
