import pandas as pd
import MeCab
import mojimoji
from tqdm import tqdm


class NaturalLang:

    def wakachi_mecab(self, _df: pd.DataFrame, _col: str):
            """
            :param _df: excelデータをDataFrameへ変換したもの。
            :return: dict
            """
            word_dict = {}
            try:
                gaiyo = _df[_col].values  # type: list
            except:
                print('[Warn] 列「' + _col + '」が見つかりません。')
                return None
            # 文字列の統一
            for g in range(len(gaiyo)):
                gaiyo[g] = mojimoji.han_to_zen(str(gaiyo[g]), kana=False, ascii=False)
                gaiyo[g] = gaiyo[g].replace('(', '（').replace(')', '）')
            # MeCabで分かち書き
            # In Ochasen
            # \t 区切り
            # [0]: 単語
            # [1]: よみ
            # [2]: 原型？
            # [3]: 品詞
            print('--- MeCab.')
            m = MeCab.Tagger('-Ochasen')
            gaiyo_cnt = 1
            for each_gaiyo in tqdm(gaiyo):
                gc = str(gaiyo_cnt).zfill(6)
                tmp = m.parse(each_gaiyo).split('\n')
                word_dict[gc] = {}
                word_cnt = 1
                for w in tmp:
                    wc = str(word_cnt).zfill(4)
                    # 'EOS'対策
                    try:
                        w.split('\t')[3]
                    except IndexError:
                        continue
                    word_dict[gc][wc] = {}
                    try:
                        # 品詞の登録 for Ochasen
                        word_dict[gc][wc]['hinshi'] = []
                        for i in range(4):
                            try:
                                word_dict[gc][wc]['hinshi'].append(w.split('\t')[3].split('-')[i])
                            except IndexError:
                                word_dict[gc][wc]['hinshi'].append('')
                        # 単語の登録 for Ochasen
                        word_dict[gc][wc]['word'] = w.split('\t')[0]
                        word_cnt += 1
                    # 'EOS'対策
                    except:
                        continue
                gaiyo_cnt += 1
            # Write csv file to check the result of wakachi-gaki.
            with open('./check_hinshi.csv', 'w') as f:
                import csv
                w = csv.writer(f)
                _for = list(word_dict.keys())
                _for.sort()
                for i in _for:
                    _tmpw = []
                    _tmph = []
                    _for2 = list(word_dict[i].keys())
                    _for2.sort()
                    for j in _for2:
                        _tmpw.append(word_dict[i][j]['word'])
                        _tmph.append('-'.join(word_dict[i][j]['hinshi']))
                    w.writerow(_tmpw)  # word
                    w.writerow(_tmph)  # 品詞
            f.close()
            return word_dict
