# -*- coding: utf-8 -*-
import jaconv
import MeCab
import os
import random
import sqlite3
import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from multiprocessing import Pool
from tqdm import tqdm


class UtilTokenizer:
    def __init__(self, mecab_option: str = ''):
        if mecab_option == '':
            self.mecab = MeCab.Tagger('-Ochasen')
        else:
            self.mecab = MeCab.Tagger('-Ochasen ' + mecab_option)

    def tfidf_tokenize(self):
        pass

    def setntencepiece_w2v_tokenize(self,
                                    sentences: list,
                                    model_path: str = 'models/pretrained/sentencepiece',
                                    typ: str = 'sum',
                                    seq_padding_typ: int = 0,
                                    seq_len: int = 512):
        """
        入力された文章を、wikipedia学習済みのsentencepiece(300次元)でベクトル化する処理.
        学習済みモデルはここから貰った.
        https://github.com/lhideki/text-vectorian/blob/master/text_vectorian/config.yml

        :param sentences: ベクトル化したい文章の配列. [setnence1, sentence2, ...]
        :param model_path: sentencepieceの学習済みモデルのPath.
        :param typ: ベクトル化の種類を指定する. sum: 単語毎のベクトルを足し上げる. seq: 時系列のベクトルにする.
        :param seq_padding_typ: typ=seqの場合の、padding方法. 0: 先頭から埋めて余ったらゼロ埋め. 1: normalを逆順にする. 2: 後ろから埋めて余ったらゼロ埋め.
        :param seq_len: typ=seqの場合の、シーケンス長.
        :return:
        """
        import sentencepiece as spm
        from gensim.models import Word2Vec

        # Load sentencepiece model
        sp = spm.SentencePieceProcessor()
        sp_model = sp.load(os.path.join(model_path, 'wikija-sentencepiece_300.model'))

        # Load word2vec model
        model = Word2Vec.load(os.path.join(model_path, 'wikija-sentencepieced_word2vec_300.model'))

        # 文章をsub-wordに分割する.
        split_sentences = []
        for _sentence in sentences:
            split_sentence = sp_model.encode_as_pieces(_sentence)  # 分割

            # 分割したsub-wordごとの処理.
            word_in_a_sentence = []
            for each_word in split_sentence:
                word_in_a_sentence.append(each_word)

            # 1文をsub-wordに分割した結果を追加.
            split_sentences.append(word_in_a_sentence)

        sentence_vectors = []

        # Vector足上げ.
        if typ == 'sum':
            for each_sentence in split_sentences:
                one_sentence_vec = np.zeros(shape=(model.wv.vector_size,))
                for each_word in each_sentence:
                    try:
                        one_sentence_vec += model.wv[each_word]
                    except:
                        print('[natural_language][sentencepiece_tokenize] Does not exist sub-word: {}'.format(each_word))
                        continue
                sentence_vectors.append(one_sentence_vec)

        # 時系列順.
        elif typ == 'seq':

            # 後方から埋める
            if seq_padding_typ == 2:
                for each_sentence in split_sentences:
                    one_sentence_seq_vec = np.zeros(shape=(seq_len, model.wv.vector_size))
                    for i, each_word in enumerate(each_sentence[::-1]):
                        try:
                            if i < seq_len:
                                print('word: {}\tvec: {}'.format(each_word, model.wv[each_word][0]))  # for debug.
                                one_sentence_seq_vec[-i-1, :] = model.wv[each_word]
                        except:
                            print('[natural_language][sentencepiece_tokenize] Does not exist sub-word: {}'.format(each_word))
                            continue
                    sentence_vectors.append(one_sentence_seq_vec)

            # 先頭から埋める
            else:
                for each_sentence in split_sentences:
                    one_sentence_seq_vec = np.zeros(shape=(seq_len, model.wv.vector_size))
                    for i, each_word in enumerate(each_sentence):
                        try:
                            if i < seq_len:
                                print('word: {}\tvec: {}'.format(each_word, model.wv[each_word][0]))  # for debug.
                                one_sentence_seq_vec[i, :] = model.wv[each_word]
                        except:
                            print('[natural_language][sentencepiece_tokenize] Does not exist sub-word: {}'.format(each_word))
                            continue
                    sentence_vectors.append(one_sentence_seq_vec)

        # エラー
        else:
            print('[natural_language][w2v_tokenize] Unknown return type: {}'.format(typ))

        # Return
        sentence_vectors = np.stack(sentence_vectors)
        if seq_padding_typ == 1:
            return np.flip(sentence_vectors, axis=1)  # seqの順番を反転させる. [2, 3, 4, 0, 0, 0] -> [0, 0, 0, 4, 3, 2]
        else:
            return sentence_vectors

    def w2v_tokenize(self,
                     sentences: list,
                     model_path: str = 'models/pretrained/word2vec/jawiki.all_vectors.100d.bin',
                     hinshi_filter: list = [],
                     typ: str = 'sum',
                     seq_padding_typ: int = 0,
                     seq_len: int = 256,
                     ):
        """
        入力された文章を、wikipedia学習済みのWord2Vecでベクトル化する処理.
        学習済みモデルはここから貰った. txtだったので、save_word2vec_format('jawiki.all_vectors.100d.bin', binary=True).
        http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/

        :param sentences: ベクトル化したい文章の配列. [setnence1, sentence2, ...]
        :param model_path: word2vecの学習済みモデルのPath. KeyedVectorsでload可能な形式.
        :param hinshi_filter: 分かち書きした際に特定の品詞だけを使いたい場合、この配列に指定する. ['名詞', '動詞']
        :param typ: ベクトル化の種類を指定する. sum: 単語毎のベクトルを足し上げる. seq: 時系列のベクトルにする.
        :param seq_padding_typ: typ=seqの場合の、padding方法. 0: 先頭から埋めて余ったらゼロ埋め. 1: normalを逆順にする. 2: 後ろから埋めて余ったらゼロ埋め.
        :param seq_len: typ=seqの場合の、シーケンス長.
        :return:
        """
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        # 文章を単語に分割する.
        wakachied_sentences = []
        for _sentence in sentences:
            wakachied_sentence = self.mecab.parse(_sentence).split('\n')[:-2]  # 分かち

            # 分かち書きした単語ごとの処理.
            word_in_a_sentence = []
            for each_word in wakachied_sentence:
                _hinshi = each_word.split('\t')[3].split('-')[0]  # 品詞: 名詞-一般名詞
                _genkei = each_word.split('\t')[2]  # 原型

                # 品詞の指定がある場合.
                if len(hinshi_filter) != 0:
                    if _hinshi in hinshi_filter:
                        word_in_a_sentence.append(_genkei)

                # 品詞の指定が無い場合.
                else:
                    word_in_a_sentence.append(_genkei)

            # 1文を分かち書きした結果を追加.
            wakachied_sentences.append(word_in_a_sentence)

        sentence_vectors = []

        # Vector足上げ.
        if typ == 'sum':
            for each_sentence in wakachied_sentences:
                one_sentence_vec = np.zeros(shape=(model.wv.vector_size,))
                for each_word in each_sentence:
                    try:
                        one_sentence_vec += model.wv[each_word]
                    except:
                        print('[natural_language][w2v_tokenize] Does not exist word: {}'.format(each_word))
                        continue
                sentence_vectors.append(one_sentence_vec)

        # 時系列順.
        elif typ == 'seq':

            # 後方から埋める
            if seq_padding_typ == 2:
                for each_sentence in wakachied_sentences:
                    one_sentence_seq_vec = np.zeros(shape=(seq_len, model.wv.vector_size))
                    for i, each_word in enumerate(each_sentence[::-1]):
                        try:
                            if i < seq_len:
                                print('word: {}\tvec: {}'.format(each_word, model.wv[each_word][0]))  # for debug.
                                one_sentence_seq_vec[-i-1, :] = model.wv[each_word]
                        except:
                            print('[natural_language][w2v_tokenize] Does not exist word: {}'.format(each_word))
                            continue
                    sentence_vectors.append(one_sentence_seq_vec)

            # 先頭から埋める
            else:
                for each_sentence in wakachied_sentences:
                    one_sentence_seq_vec = np.zeros(shape=(seq_len, model.wv.vector_size))
                    for i, each_word in enumerate(each_sentence):
                        try:
                            if i < seq_len:
                                one_sentence_seq_vec[i, :] = model.wv[each_word]
                        except:
                            print('[natural_language][w2v_tokenize] Does not exist word: {}'.format(each_word))
                            continue
                    sentence_vectors.append(one_sentence_seq_vec)

        # エラー
        else:
            print('[natural_language][w2v_tokenize] Unknown return type: {}'.format(typ))

        # Return
        sentence_vectors = np.stack(sentence_vectors)
        if seq_padding_typ == 1:
            return np.flip(sentence_vectors, axis=1)  # seqの順番を反転させる. [2, 3, 4, 0, 0, 0] -> [0, 0, 0, 4, 3, 2]
        else:
            return sentence_vectors

    def fasttext_tokenize(self):
        pass

    def glove_tokenize(self):
        pass

    def elmo_tokenize(self):
        pass


class UtilAugmentation:
    """
    テキストデータのデータ拡張を行う機能.
    以下の論文を参考に実装.

    EDA: Easy Data Augmentation Techniques for
    Boosting Performance on Text Classification Tasks
    https://arxiv.org/abs/1901.11196

    CutMix: Regularization Strategy to Train
    Strong Classifiers with Localizable Features
    https://arxiv.org/abs/1905.04899
    """
    def __init__(self,
                 model_path: str = 'models/pretrained/word2vec/jawiki.all_vectors.100d.bin',
                 wordnet_path: str = 'models/nlp/wordnet/wnjpn.db'
                 ):
        from gensim.models import KeyedVectors
        self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.mecab = MeCab.Tagger('-Ochasen')
        self.conn = sqlite3.connect(wordnet_path)

    def get_synonym_word_w2v(self, _word):
        try:
            candidate_words = self.w2v_model.most_similar(_word, topn=5)
            pickup_idx = int(random.uniform(0, 4))
        except:
            return _word
        return candidate_words[pickup_idx][0]

    def get_synonym_word_wordnet(self, _word):
        # WordNetに存在する単語か検索
        cur = self.conn.execute("select * from word where lemma=?", (_word,))
        word_list = [row for row in cur]
        if len(word_list) == 0:
            return _word

        # 類義語を検索
        synonym_list = []
        for word in word_list:
            cur = self.conn.execute("select * from sense where wordid=?", (word[0],))
            synnet_list = [row for row in cur]
            for synnet in synnet_list:
                cur = self.conn.execute(
                    "select * from sense, word where synset = ? and "
                    "word.lang = 'jpn' and "
                    "sense.wordid = word.wordid;",
                    (synnet[0],)
                )
            synonym_list += [row[9] for row in cur]

        # 候補からランダムに選択
        return synonym_list[int(random.uniform(0, len(synonym_list)))]

    def wakachi(self, _sentence):
        return self.mecab.parse(_sentence).split('\n')[:-2]

    def synonym_replacement(self,
                            _sentences: list,
                            _replacement_type: str = 'wordnet',
                            _replacement_prob: float = 0.5,
                            _replacement_hinshi=None,
                            _return_with_wakachi: bool = True,
                            _num_of_argment: int = 10,
                            ):
        """
        word2vecやwordnetを活用し、文書の単語を類義語へ置換し、Data Augmentationする機能.

        :param _sentences: 水増ししたい文章のリスト. e.g. [sentence1, sentence2, ...]
        :param _replacement_type: 類義語の検索方法. wordnet or w2v
        :param _replacement_prob: 置換対象の単語を置換する確率.
        :param _replacement_hinshi: 置換対象の単語の品詞.
        :param _return_with_wakachi: 分かち書きされた形式で返すかどうか.
        :param _num_of_augment: 　1文書を水増しする試行数.
        :return:
        """
        # Default list の値設定.
        # PyCharm先生にこうしろと言われた.
        if _replacement_hinshi is None:
            _replacement_hinshi = ['形容詞']

        replaced_sentences = []

        # 文章ごとに類似語に置換.
        for each_sentence in _sentences:
            words = self.wakachi(each_sentence)

            # 水増ししたい回数だけ実施.
            for _ in range(_num_of_argment):
                replaced_obj = []

                # 文章内の単語毎に類義語を検索.
                for idx, each_word in enumerate(words):

                    # 置換対象の単語
                    if each_word.split('\t')[3].split('-')[0] in _replacement_hinshi and random.random() <= _replacement_prob:  # 置換の判定

                        # WordNet
                        if _replacement_type == 'wordnet':
                            synonym_words = self.get_synonym_word_wordnet(_word=each_word.split('\t')[2])
                            if synonym_words == each_word.split('\t')[2]:  # wordnetで見つからない単語は、
                                synonym_words = each_word.split('\t')[0]

                        # Word2Vec
                        else:
                            synonym_words = self.get_synonym_word_w2v(_word=each_word.split('\t')[0])
                        replaced_obj.append(synonym_words)

                    # 置換対象外の単語
                    else:
                        replaced_obj.append(each_word.split('\t')[0])

                # 置換した文書をlistへ追加.
                if _return_with_wakachi:
                    replaced_sentences.append(replaced_obj)  # 分かち書きで返す
                else:
                    replaced_sentences.append(''.join(replaced_obj))  # 文章で返す

        return replaced_sentences

    def random_insertion(self):
        pass

    def random_swap(self):
        pass

    def random_deletion(self):
        pass

    def cut_mix(self):
        pass


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
            each_item[g] = jaconv.h2z(str(each_item[g]), kana=False, ascii=False)
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
                gaiyo[g] = jaconv.h2z(str(gaiyo[g]), kana=False, ascii=False)  # 半角→全角
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
    def __init__(self, _sentences, _model=None):
        """
        :param _sentences: 学習に使うコーパス [['I', 'like', 'python'], ['You', 'like', 'python']...]
        """
        self.sentences = _sentences
        self.w2v_model = _model

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
    def __init__(self, _sentences, _model=None):
        self.d2v_model = _model
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
            print('[D2V] model is None.')
            return None
        self.d2v_model.save(_save_path)

    def load_model(self, _load_path):
        self.d2v_model = Doc2Vec.load(_load_path)

    def search_sim_word(self, word1, word2):
        if self.d2v_model is None:
            print('[D2V] model is None.')
            return None
        return self.d2v_model.similarity(word1, word2)


if __name__ == '__main__':
    # tokenizer = UtilTokenizer()
    # ret = tokenizer.w2v_tokenize(sentences=['今日は寝過ごした。', '昼寝は無いだろう。'],
    #                              typ='seq',
    #                              seq_padding_typ=2,
    #                              seq_len=5)
    aug = UtilAugmentation()
    ret = aug.synonym_replacement(['今日は面白いくらいに寝過ごした。', '昼寝は決して行わないだろう。'],
                                  _replacement_prob=1,
                                  _replacement_hinshi=['動詞', '名詞', '形容詞'],
                                  _return_with_wakachi=False,
                                  )
    print(ret)
