import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras_self_attention import SeqSelfAttention
from keras import Sequential, Input, Model
from keras.layers import Dense, LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class Util:
    def __init__(self):
        pass

    @staticmethod
    def colorize_attention(words, attention_weight):
        """
        Attention weightの高い単語を視覚的に確認するための機能.

        :param words: 分割された単語リスト.
        :param attention_weight: Attention weight.
        :return: html形式のテキスト（色でattention weightを確認可能)
        """
        cmap = matplotlib.cm.get_cmap('Reds')
        template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
        colored_string = ''
        for word, color in zip(words, attention_weight):
            color = matplotlib.colors.rgb2hex(cmap(color)[:3])
            colored_string += template.format(color, word)

        return colored_string


class ImageRecognition:

    def __init__(self):
        self.optimizer = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.train_generator = None
        self.validation_generator = None

    def import_data(self, _x, _y):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(_x.values, _y.values, test_size=0.25)

    def import_data_from_dir(self, _train_dir: str, _valid_dir: str, _img_size: tuple):
        self.train_generator = ImageDataGenerator().flow_from_directory(
            directory=_train_dir,
            target_size=_img_size,
            batch_size=30,
            class_mode='categorical')

        self.validation_generator = ImageDataGenerator().flow_from_directory(
            directory=_valid_dir,
            target_size=_img_size,
            batch_size=1,
            class_mode='categorical')

    def define_optimizer(self):
        self.optimizer = Adam(lr=0.001)

    def execute_keras_model(self, _type: str,
                            _model: Sequential,
                            _optimizer=None,
                            _batch_size: int = 64,
                            _epoch: int = 5):
        """
        受け取ったモデルで学習を実行する機能
        :param _type: 'category' or 'regression'
        :param _model: Keras model.
        :param _optimizer: Keras Optimizer.
        :param _batch_size: Batch Size. Default: 64
        :param _epoch: Epoch. Default: 5
        :return: None as Error. 0 as Success.
        """
        #Train and Valid data check.
        # if (self.x_train is None or self.y_train is None) or self.train_generator is None: return None
        # if (self.x_valid is None or self.y_valid is None) or self.validation_generator is None: return None
        train_data_type = None
        if self.x_train is not None:
            train_data_type = 'from_list'
        elif self.train_generator is not None:
            train_data_type = 'from_dir'
        else:
            return train_data_type

        # Optimizer check.
        if _optimizer is None:
            self.define_optimizer()
        else:
            self.optimizer = _optimizer

        # Compile model.
        if _type == 'category':
            _model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        elif _type == 'regression':
            _model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['accuracy'])

        if train_data_type == 'from_list':
            history = _model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=_batch_size,
                                 epochs=_epoch,
                                 validation_data=(self.x_valid, self.y_valid))
        elif train_data_type == 'from_dir':
            history = _model.fit_generator(self.train_generator,
                                           epochs=_epoch,
                                           validation_data=self.validation_generator,
                                           shuffle=True,
                                           verbose=1)
        else:
            print('[Warn] Invalid type.')
            return None

        # Save graph of loss.
        plt.figure(figsize=(8, 5))
        plt.ylabel('val_loss')
        plt.xlabel('epochs')
        plt.tight_layout()
        plt.plot(history.history['val_loss'], "o-", markersize=2, label='val_loss')
        plt.plot(history.history['loss'], "o-", markersize=2, label='loss')
        plt.legend()
        plt.savefig(os.path.join('..', 'loss.png'))

        # Save graph of accuracy.
        plt.figure(figsize=(8, 5))
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.tight_layout()
        plt.plot(history.history['val_acc'], "o-", markersize=2, label='val_acc')
        plt.plot(history.history['acc'], "o-", markersize=2, label='acc')
        plt.legend()
        plt.savefig(os.path.join('..', 'acc.png'))

        return 0


class TextClassification:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 seq_len: int = 512,
                 seq_dim: int = 100,
                 h_dim: int = 300,
                 epochs: int = 100,
                 ):
        """
        簡易的に文書分類を行うクラス.
        Self attention + Bi-LSTM.

        :param x: 学習データ. shape: (batch_size, seq_len, seq_dim)
        :param y: 正解ラベル. shape: (batch_size,)
        :param seq_len: 時系列長
        :param seq_dim: 各sequenceの次元
        :param h_dim: Bi-LSTMの隠れ層次元
        :param epochs: 学習Epoch
        """
        import MeCab
        self.mecab = MeCab.Tagger('-Owakati')

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.h_dim = h_dim
        self.epochs = epochs

        self.load_data(x, y)
        self.build_model()

    def load_data(self, _x, _label):
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        # X
        self.x = _x

        # Y
        le = preprocessing.LabelEncoder()
        le.fit(_label)
        label_vecs = le.transform(_label)
        label_vecs = label_vecs.reshape(len(label_vecs), 1)
        self.y = preprocessing.OneHotEncoder(sparse=False).fit_transform(label_vecs)

        # Split data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x,
                                                                                self.y,
                                                                                test_size=0.33,
                                                                                )

    def build_model(self):
        """
        self attention + bi-lstmモデルを作って学習する.
        :return:
        """
        # Build model.
        inp = Input(batch_shape=[None, self.seq_len, self.seq_dim])
        att_layer, attention_weight = SeqSelfAttention(return_attention=True)(inp)
        out = Bidirectional(LSTM(self.h_dim))(att_layer)
        output = Dense(len(self.train_y[0]), activation='softmax')(out)
        self.model = Model(inp, output)

        # Compilie and train
        self.model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['acc', 'mse', 'mae'])
        self.model.summary()
        self.model.fit(self.train_x,
                       self.train_y,
                       epochs=self.epochs,
                       batch_size=20,
                       validation_data=(self.test_x, self.test_y),
                       )

        # Extract self attention.
        self.model_for_attention_visualise = Model(inp, attention_weight)

    def visualise_attention_weight(self, predict_str, predict_data):
        """
        Attention weightを視覚化する処理.

        :param predict_str: 視覚化したい文を分かち書きしたもの
        :param predict_data: 視覚化したいデータ. shape: (1, seq_len, seq_dim)
        :return:
        """
        weight = self.model_for_attention_visualise.predict(predict_data)
        weight = [w.max() for w in weight[0, :, :]]
        weight_M = [x / max(weight) for x in weight]

        colored_str = Util().colorize_attention(predict_str, weight_M)

        return colored_str
