import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy import fftpack
from . import plot


class DataManipulation:
    def __init__(self):
        self.p = plot.Plot()

    def ica_reduction(self, _df: pd.DataFrame, _dim: int, _return_with_model=False):
        """
        独立成分分析でノイズを除去するもの。
        :param _df: 次元削減したいデータ(DataFrame形式)
        :param _dim: 削減後の次元数
        :param _return_with_model: 主成分分析のモデルもreturnするかどうかのフラグ
        :return: 主成分分析後のデータフレーム(ndarray)
        """
        ica = FastICA(n_components=_dim)  # type: FastICA
        M = np.mean(_df.values, axis=1)[:, np.newaxis]  # type: np.ndarray
        norm_ary = _df.values - M  # type: np.ndarray
        ica.fit(norm_ary)
        transformed = ica.transform(norm_ary)
        if _return_with_model:
            return transformed, ica
        else:
            return transformed

    def pca_reduction(self, _df: pd.DataFrame, _dim: int, _return_with_model=False):
        """
        主成分分析で次元を削減するもの。
        :param _df: 次元削減したいデータ(DataFrame形式)
        :param _dim: 削減後の次元数
        :param _return_with_model: 主成分分析のモデルもreturnするかどうかのフラグ
        :return: 主成分分析後のデータフレーム(ndarray)
        """
        pca = PCA(n_components=_dim)  # type: PCA
        pca.fit(_df.values)
        transformed = pca.fit_transform(_df.values)  # type: np.ndarray
        if _return_with_model:
            return transformed, pca
        else:
            return transformed

    def simple_fft(self, _input_file: str):
        """
        入力値を離散フーリエ変換し、周波数成分に分解する機能
        :param _input_file:
        :return:
        """
        import matplotlib.pyplot as plt
        (time, data) = np.loadtxt(_input_file, unpack=True, delimiter=",",
                                  usecols=(0, 1))  # type: (np.ndarray, np.ndarray)
        sample_freq = fftpack.fftfreq(data[:].size, d=1)
        y_fft = fftpack.fft(data[:])
        pidxs = np.where(sample_freq > 0)
        freqs, power = sample_freq[pidxs], np.abs(y_fft)[pidxs]
        plt.figure()
        plt.plot(freqs, power, 'b-')
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Power")
        plt.show()

    def fill_na(self, _df: pd.DataFrame, _typ: int):
        """
        fill na to all columns with previous value.
        :param _df: NAを埋めたいData Frame.
        :param _typ: NAの埋め方（0: 前行値で補完、1: 線形補完）
        :return:  NAを前行の値で埋めたDataFrame.
        """
        _df_isnull = _df.isnull()
        # 前行補完
        if _typ == 0:
            for i in range(len(_df)):
                for j in range(len(_df.index)):
                    if _df_isnull.iat[j, i]:
                        _df.replace(_df.iat[j, i], _df.iat[j - 1, i], inplace=True)
            return _df
        # 線形補完
        elif _typ == 1:
            return _df.interpolate()

    def drop_na(self, _df: pd.DataFrame):
        """
        NAが含まれる行を削除するもの。
        :param _df: NAが含まれる行を削除したいデータ(DataFrame形式)
        :return: NAが削除された後のDataFrame
        """
        return _df.dropna(how='any')
