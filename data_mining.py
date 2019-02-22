import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
        :return: 主成分分析後のndarray
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

    def lda_reduction(self, _df: pd.DataFrame, _x: int, _y: int, _dim: int, _return_with_model=False):
        """
        LDAによる教師有データの次元削減を行う機能
        :param _df: 次元削減したいデータ(DataFrame形式)
        :param _x: 学習データ
        :param _y: 正解データ
        :param _dim: 削減後の次元数
        :param _return_with_model: 主成分分析のモデルもreturnするかどうかのフラグ
        :return:
        """
        lda = LDA(n_components=_dim)  # type: LDA
        lda.fit(_df.iloc[:, _x].values, _df.iloc[:, _y].values)
        transformed = lda.fit_transform(_df.values)  # type: np.ndarray
        if _return_with_model:
            return transformed, lda
        else:
            return transformed

    def tsne_reduction(self, _dim: int, _x, _return_with_model: bool):
        """
        t-SNEによる次元削減を行う機能
        :param _dim: 削減後の次元数（t-SNEの場合、2or3次元）
        :param _x: 次元削減したいデータ（DataFrame形式）
        :param _return_with_model:
        :return:
        """
        if _dim > 3:
            print('[Warn][tsne_reduction] It is to be desired that the dimention is 2 or 3.')
        tsne = TSNE(n_components=_dim)
        transformed = tsne.fit_transform(_x)
        if _return_with_model:
            return transformed, tsne
        else:
            return transformed

    def simple_fft(self, _input_file: str):
        """
        入力値を離散フーリエ変換し、周波数成分に分解する機能
        :param _input_file: 変換したいcsvファイル
        :return: None
        """
        # Todo: UtilToolsに合うように、_dfインプットからのpure2dplot連携を実現したい
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

    def crosstabs(self, _df: pd.DataFrame, _x: int, _y: int, _val: int = None):
        return pd.crosstab(_df.iloc[:, _x], _df.iloc[:, _y])


from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser


class GetYUHOElementInfo:
    """
    有価証券報告のXBRLファイルから
    必要なラベルの情報だけを取得するクラス
    Thanks a lot to EdinetXbrlParser !!
    SUPER Useful !!
    """

    def __init__(self):
        """
        Initialize
        """
        self.parser = EdinetXbrlParser()
        self.edinet_xbrl_object = None
        self.businessRisks = None

    def get_taxonomy_data(self):
        """
        タクソノミデータを取得する処理
        :return:
        """
        pass

    def get_required_element(self):
        """
        「XBRLから取得したい要素」を取得する処理
        :return:
        """
        pass

    def parse_xbrl(self, _xbrl_path: str):
        """
        Xbrlファイルから何年度のデータかを判別した上で、parseする処理
        :param _xbrl_path:
        :return:
        """
        # Todo: 読み込むxbrlファイルから、何年度のデータなのかを判別し、適切なタクソノミ情報と突き合わせるコードを書くこと.
        self.edinet_xbrl_object = self.parser.parse_file(_xbrl_path)

    def get_element_data(self, _key: str, _contextRef: str):
        """
        必要な要素を抜き出す処理
        :param _key:
        :param _contextRef:
        :return:
        """
        if _contextRef == '':
            self.businessRisks = self.edinet_xbrl_object.get_data_list(_key)
        else:
            self.businessRisks = self.edinet_xbrl_object.get_data_by_context_ref(_key, _contextRef)

    def cleansing(self):
        """
        htmlタグを除去する処理.
        :return:
        """
        # Todo: 取得した要素の中のタグ情報を除去する機能を書くこと.
        pass

    def export_data(self):
        """
        CSV形式でデータをエクスポートする処理.
        :return:
        """
        pass


if __name__ == '__main__':
    ufo = GetYUHOElementInfo()
    # Todo: 企業ごとにxbrlを取得するコードを書くこと.
    # Todo: 欲しい要素を指定して、それだけ取得する処理もほしい.なんか楽して取れる方法ないかな...
    xbrl_file_path = r""
    key = 'jpcrp_cor:BusinessRisksTextBlock'  # 企業等のリスク
    print(businessRisks[0].get_value())
