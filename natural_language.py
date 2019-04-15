# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import MeCab
import os
import mojimoji
from gensim.models import word2vec
from tqdm import tqdm
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class NaturalLang:
    def __init__(self, num_to_exec_multithread: int = 3):
        self.num_to_exec_multithread = num_to_exec_multithread
        self.wakachi_result_for_multiprocess = {}
        pass

    def do_wakachi(self, args: list):
        """
        分かち書きの処理を行う機能(multi_threadとして呼び出される側の関数)
        :param args: 引数を纏めたリスト[0]: MeCabのオプション、[1]: 分かち書き結果のdict変数の開始キー、[2]: 分かち書き対象のSeries.
        :return:
        """
        _out_num = args[0]
        _option = args[1]
        _item_count = args[2]
        df = args[3]
        _col = args[4]
        word_dict = {}
        try:
            each_item = df[_col].values  # type: np.ndarray
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
        option = '-Ochasen' if _option is None else '-Ochasen ' + _option
        m = MeCab.Tagger(option)
        item_cnt = _item_count
        print('start')
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
        return [_out_num, word_dict]

    def wakachi_mecab(self,
                      _df: pd.DataFrame,
                      _col: str, _export_result_wakachi: bool = False,
                      _export_file_path: str='./export_result_wakachi.csv',
                      _option: str = None):
        """
        :param _df: excelデータをDataFrameへ変換したもの。
        :param _col: _df内の、分かち書きをしたい列名
        :param _export_result_wakachi: 分かち書きした結果を品詞とともにcsvにエクスポートするかどうか（default: False）
        :param _export_file_path: file path to be exported.
        :param _option: MeCab option you wanna use.
        :return: dict
        """
        # 分かち書きするデータ数が100未満なら、シングルコアで処理する
        total_num_of_exec = len(_df)
        if total_num_of_exec < 100:
            word_dict = self.do_wakachi(args=[0, _option, 1, _df, _col])[1]
        # multiprocess で分かち書き
        else:
            word_dict = {}
            num_of_thread = 3
            p = Pool(num_of_thread)
            q, mod = divmod(len(_df), num_of_thread)
            args = []
            for i in range(num_of_thread):
                if i == num_of_thread - 1:
                    args.append([i, _option, q * i + 1, _df.iloc[q*i:q*(i + 1)+mod, :].reset_index(drop=True), _col])
                else:
                    args.append([i, _option, q*i+1, _df.iloc[q*i:q*(i+1), :].reset_index(drop=True), _col])
            word_dicts = p.map(self.do_wakachi, args)
            p.close()
            for wd in word_dicts:
                self.wakachi_result_for_multiprocess[wd[0]] = wd[1]
            for i in range(num_of_thread):
                word_dict.update(self.wakachi_result_for_multiprocess[i])

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

    def create_histogram(self, _df: pd.DataFrame, _column: str, _typ: str, _ext_hinshi: list=['一般', '固有名詞'], _op = None):
            """
            ヒストグラムを作成する。
            :param _df:
            :param _column:
            :param _typ: meishi, or all
            :param _ext_hinshi:
            :param _op: MeCab option if needed.
            :return:
            """
            import collections
            _word_box = []
            count = 0
            print('[Info] Start to import file.')
            gaiyo = _df[_column].values  # type: np.ndarray
            # 文字列の統一
            for g in range(len(gaiyo)):
                import mojimoji
                gaiyo[g] = mojimoji.han_to_zen(str(gaiyo[g]), kana=False, ascii=False)  # 半角→全角
                gaiyo[g] = gaiyo[g].replace('(', '（').replace(')', '）')  # 半角カッコを全角に統一する
            print('[Info] Start to create histogram.')
            if _op is not None:
                m = MeCab.Tagger(_op)
            else:
                m = MeCab.Tagger()
            for i in tqdm(gaiyo):
                count += len(i.split(' '))
                if _typ == 'meishi':
                    _word_info = i.split(' ')
                    for _each_word in _word_info:
                        if _each_word == '': continue
                        _w = m.parse(_each_word).split('\n')[0:-2]
                        hinshi = _w[0].split('\t')[1]
                        if hinshi.split(',')[0] == '名詞':
                            type1 = hinshi.split(',')[1]
                            if type1 in _ext_hinshi:
                                _word_box.append(_w[0].split('\t')[0])
                elif _typ == 'all':
                    _word_info = i.split(' ')  # MeCab default dic
                    _word_box.extend(_word_info)
                else:
                    print('invalid type.')

            cnt = collections.Counter(_word_box)
            # count words.
            cnt_df = pd.DataFrame.from_dict(cnt, orient='index')  # convert Counter to DataFrame
            cnt_df = cnt_df.rename(columns={0: 'count'})  # rename columns
            cnt_df = cnt_df.sort_values('count', ascending=False)  # sort in descending order
            f_brackets = lambda x: x / count
            cnt_df_2 = cnt_df.iloc[:, 0].map(f_brackets)
            output_df = pd.concat([cnt_df, cnt_df_2], axis=1)
            return output_df

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

    def bm25(self, _corpus: list):
        from gensim.summarization.bm25 import get_bm25_weights
        result = get_bm25_weights(_corpus)



class W2V:
    def __init__(self, _sentences):
        """
        :param _sentences: 学習に使うコーパス [['I', 'like', 'python'], ['You', 'like', 'python']...]
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


class D2V:
    def __init__(self, _sentences):
        self.d2v_model = None
        self.sentences = [TaggedDocument(doc, [i]) for i, doc in enumerate(_sentences)]

    def create_model(self, _dm: int=1, _size: int=300, _window: int=8, _min_count: int=10, _workers: int=4):
        """
        Doc2Vecのモデルを生成する機能
        :param _dm: training algorithm. 1 means 'distributed memory'. otherwise, 'distributed bag-of-words'
        :param _size: vector size of model.
        :param _window:
        :param _min_count:
        :param _workers: Use these many worker threads to train the model (=faster training with multi-core machines).
        :return:
        """
        self.d2v_model = Doc2Vec(documents=self.sentences,
                                 dm=_dm,
                                 size=_size,
                                 window=_window,
                                 alpha=0.025,
                                 min_alpha=0.025,
                                 min_count=_min_count,
                                 workers=_workers,
                                 compute_loss=True)  # type: Doc2Vec

    def train(self, _epoch: int):
        """
        create_modelで生成したモデルを学習させる機能 loss計算機能はまだないらしい
        :param _epoch:
        :return:
        """
        if self.d2v_model is None:
            print('[D2V] model is empty.')
            return None
        for i in tqdm(range(_epoch)):
            print('now alpha : ' + str(self.d2v_model.alpha))
            self.d2v_model.train(self.sentences,
                                 total_examples=len(self.sentences),
                                 epochs=1)
            self.d2v_model.alpha -= (0.025 - 0.0001)/_epoch

    def save_model(self, _save_path):
        """
        Doc2Vecのモデルを保存する処理。
        :param _save_path: 保存するファイルパス
        :return:
        """
        if self.d2v_model is None:
            print('[D2V] model is empty.')
            return None
        self.d2v_model.save(_save_path)

    def load_model(self, _load_path):
        self.d2v_model = Doc2Vec.load(_load_path)

    def search_sim_word(self, word1, word2):
        if self.d2v_model is None:
            print('[D2V] model is empty.')
            return None
        return self.d2v_model.similarity(word1, word2)
