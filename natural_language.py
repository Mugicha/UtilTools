# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import MeCab
import os
import mojimoji
from gensim.models import word2vec
from tqdm import tqdm


class NaturalLang:
    def __init__(self):
        pass

    def wakachi_mecab(self, _df: pd.DataFrame, _col: str, _export_result_wakachi: bool = False, _export_file_path: str='./export_result_wakachi.csv'):
        """
        :param _df: excelデータをDataFrameへ変換したもの。
        :param _col: _df内の、分かち書きをしたい列名
        :param _export_result_wakachi: 分かち書きした結果を品詞とともにcsvにエクスポートするかどうか（default: False）
        :return: dict
        """
        word_dict = {}
        try:
            each_item = _df[_col].values  # type: np.ndarray
        except:
            print('[Warn][wakachi_mecab] 列「' + _col + '」が見つかりません。')
            return None
        # 文字列の統一
        for g in range(len(each_item)):
            each_item[g] = mojimoji.han_to_zen(str(each_item[g]), kana=False, ascii=False)
            each_item[g] = each_item[g].replace('(', '（').replace(')', '）')
        # MeCabで分かち書き
        # In Ochasen
        # \t 区切り
        # [0]: 単語
        # [1]: よみ
        # [2]: 原型？
        # [3]: 品詞
        m = MeCab.Tagger('-Ochasen')
        item_cnt = 1
        for each_sentence in tqdm(each_item):
            gc = str(item_cnt).zfill(6)
            tmp = m.parse(each_sentence).split('\n')
            word_dict[gc] = {}
            word_cnt = 1
            for w in tmp:
                wc = str(word_cnt).zfill(4)
                # 'EOS'対策
                try:
                    w.split('\t')[3]  # EOSだと4要素目は無いのでExceptionが起こる
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
            item_cnt += 1
        # Write csv file to check the result of wakachi-gaki.
        if _export_result_wakachi:
            with open(_export_file_path, 'w') as f:
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

    def knp_parsing(self, _sentences: str):
        from pyknp import KNP
        k = KNP(option='-tab', jumanpp=False)
        k.parse(_sentences)

    def tfidf(self, _corpus: list, _min_df: float = 0.03):
        """
        TF-IDFによって、特徴語を抽出する機能.
        :param _corpus: インプットとなるコーパス（type: [ [I like vim] [I like python] ... ]）
        :param _min_df: 結果として表示するidf値の最小値（default: 0.03）
        :return:
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=_min_df)
        tfidf_x = vectorizer.fit_transform(_corpus).toarray()  # type: np.ndarray
        feature_names = np.array(vectorizer.get_feature_names())
        index = tfidf_x.argsort(axis=1)[:, ::-1]  # np.ndarray
        feature_words = [feature_names[doc] for doc in index]  # type: list
        return feature_words  # type: list

class W2V:
    def __init__(self, _sentences):
        """
        :param _sentences: 学習に使うコーパス
        """
        self.w2v_model = None
        self.sentences = _sentences

    def create_model(self, _size: int, _min_cnt: int, _window: int):
        """
        create word2vec model.
        :param _size: 構築する単語ベクトルのサイズ
        :param _min_cnt: ベクトルを作成する単語の最小出現回数
        :param _window: skip-gramで使うウィンドウサイズ
        :return:
        """
        self.w2v_model = word2vec.Word2Vec(sentences=self.sentences,
                                           size=_size, min_count=_min_cnt,
                                           window=_window,
                                           compute_loss=True)  # type: word2vec.Word2Vec

    def train(self, _epoch: int, _alpha: float = 0.0001, _return_loss: bool = False):
        """
        W2Vモデルを学習させる。
        :param _epoch: エポック数（ここではイテレーションの回数）
        :param _alpha: 学習率
        :param _return_loss: エポック毎の損失関数値をリストで返すかどうか
        :return:
        """
        loss = []  # type: list
        for i in range(_epoch):
            self.w2v_model.train(self.sentences,
                                 total_examples=len(self.sentences),
                                 end_alpha=_alpha,
                                 epochs=1,
                                 compute_loss=True)
            loss.append([i, self.w2v_model.get_latest_training_loss()])
        if _return_loss:
            return loss

    def save_model(self, _output_folder: str = './', _output_model_name: str = 'w2v_model.model'):
        """
        作成したモデルを保存するメソッド
        :param _output_folder: 保存先フォルダパス
        :param _output_model_name: 保存する際のモデル名（default: w2v_model.model）
        :return:
        """
        if self.w2v_model is None:
            print("[save_model] Model you wanna save is None.")
            return
        self.w2v_model.save(os.path.join(_output_folder, _output_model_name))

    def search_sim_word(self, _word: str, topn: int = 20):
        """
        類似語を検索し、返す機能。
        :param _word:
        :param topn:
        :return:
        """
        if self.w2v_model is None:
            print("[save_model] W2V model is None.")
            return None
        return self.w2v_model.most_similar(positive=_word, topn=topn)  # type: list
