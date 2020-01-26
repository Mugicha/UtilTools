import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import fftpack
from tqdm import tqdm
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

    def moving_avg(self, _df: pd.DataFrame, _column: str, _typ: str = 'behind', _window: int = 3):
        """
        移動平均を算出し、列を追加して返す処理.
        :param _df:
        :param _column:
        :param _typ: "behind" or "center". "behind" is default value.
        :param _window:
        :return:
        """
        if _typ == 'behind':
            ma_series = _df[_column].rolling(_window).mean()
        elif _typ == 'center':
            if _window % 2 == 1:
                _tmp_ma_ndary = _df[_column].values
                ave = np.convolve(_tmp_ma_ndary, np.ones(_window) / float(_window), 'same')
                margin = _window//2
                ave[:margin] = np.nan
                ave[-1 * margin:] = np.nan
                ma_series = pd.Series(ave)
            else:
                print('[Warn] Window size must be Odd number.')
                return None
        else:
            print('[Warn] _typ is invalid.')
            return None
        _df.reset_index(drop=True, inplace=True)
        concat_df = pd.concat([_df, pd.DataFrame(ma_series.T)], axis=1)
        new_col = list(_df.columns)
        new_col.append(_column + '_ma')
        concat_df.columns = new_col
        return concat_df


class FeatureSelection:
    def __init__(self):
        pass

    @staticmethod
    def col_based_feature_selection(_df: pd.DataFrame,
                                    _cor_threshold: float=0.95,
                                    _cor_matrix: pd.DataFrame = None,
                                    _drop: bool=True,
                                    _return_col: bool=False):
        """
        相関係数の高い変数同士を探して、次元削減を行う処理
        :param _df:
        :param _cor_threshold:
        :param _cor_matrix:
        :param _drop:
        :param _return_col:
        :return:
        """
        bef_var = len(_df.columns)
        feat_corr = set()
        if _cor_matrix is None:
            _cor_matrix = _df.corr()
        for i in tqdm(range(len(_cor_matrix.columns))):
            for j in range(i):
                if abs(_cor_matrix.iloc[i, j]) > _cor_threshold:
                    feat_name = _cor_matrix.columns[i]
                    feat_corr.add(feat_name)
        if _drop:
            _df.drop(labels=feat_corr, axis='columns', inplace=True)

        print('Feature variables reduced to ' + str(bef_var - len(feat_corr)) + ' from ' + str(bef_var) + '.')
        if _return_col:
            return _df, list(feat_corr)
        else:
            return _df

    @staticmethod
    def lasso_based_feature_selection(_x, _y, _alpha: float=1.0, _iter: int=10000):
        """
        Lasso回帰(L1正則化)による次元削減
        :param _x:
        :param _y:
        :param _alpha:
        :param _iter:
        :return:
        """
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=_alpha, normalize=True, max_iter=_iter)
        lasso.fit(_x, _y)
        return lasso.coef_
