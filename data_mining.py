import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataManipulation():

    def pca_reduction(self, _df: pd.DataFrame, _dim: int, _return_with_model=False):
        """
        主成分分析で次元を削減するもの。
        :param _df: 次元削減したいデータ(DataFrame形式)
        :param _dim: 削減後の次元数
        :param _return_with_model: 主成分分析のモデルもreturnするかどうかのフラグ
        :return: 主成分分析後のデータフレーム(DataFrame)
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=_dim)  # type: PCA
        pca.fit(_df.values)
        transformed = pca.fit_transform(_df.values)  # type: np.ndarray
        if _return_with_model:
            return transformed, pca
        else:
            return transformed

    def fill_na(self, _df: pd.DataFrame, _typ: int):
        """
        fill na with previous value.
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
