import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import os


class Plot():

    def pure_2d_plot(self, _df: pd.DataFrame, _x: int, _y: int):
        """
        DataFrame内の2列を使って二次元プロットを行う機能。
        :param _df: 使用するDataFrame
        :param _x: x軸のラベル番号
        :param _y: y軸のラベル番号
        :return: matplotlib.pyplotクラス
        """
        plt.figure(figsize=(8, 6))  # plot size
        plt.plot(_df.iloc[:, _x].values, _df.iloc[:, _y].values)
        return plt

    def corr_map(self, _df: pd.DataFrame, _path='./'):
        """
        各変数の相関係数のヒートマップ画像を保存する機能。
        :param _df: ヒートマップの基となるDataFrame.
        :param _path: 保存するフォルダのパス
        :return: None.
        """
        plt.figure(figsize=(8, 6))  # heat map size
        sns.heatmap(_df.corr(), annot=True, cmap='plasma', linewidths=.5)
        plt.savefig(os.path.join(_path, 'corr_map.png'))

    def scatter_with_histogram(self, _df: pd.DataFrame, dim_reduction=False, _path='./'):
        """
        ヒストグラム付き散布図の画像を保存する機能。
        :param _df: 散布図の基となるDataFrame.
        :param dim_reduction: 次元削減するかどうか(default=False)
        :param _path: 保存するフォルダのパス
        :return: None.
        """
        plt.figure(figsize=(8, 6))
        if dim_reduction:
            from UtilTools.data_mining import DataManipulation
            pca = DataManipulation()
            _trn = pd.DataFrame(pca.pca_reduction(_df, 2, False))
            sns.jointplot(0, 1, _trn, kind='scatter')
            plt.savefig('scatter_with_histogram_pca.png')
            return None
        sns.jointplot(0, 1, _df, kind='scatter')
        plt.savefig(os.path.join(_path, 'scatter_with_histogram.png'))

    def pair_plot(self, _df: pd.DataFrame, _path='./'):
        """
        seabornのPariplot画像を保存する機能。
        :param _df: pairplotしたいDataFrame
        :param _path: 保存するフォルダのパス
        :return: None.
        """
        plt.figure(figsize=(8, 6))
        sns.pairplot(_df)
        plt.savefig(os.path.join(_path, 'pair_plot.png'))

    def scatter_emphasis(self, _df: pd.DataFrame, _x: int, _y: int, _c: int, _path='./', _s=2):
        """
        ラベルによって色を変えて散布図を表示する機能。
        :param _df: 表示させたいDataFrame
        :param _x: x軸の要素
        :param _y: y軸の要素
        :param _c: 色を変えるラベルとなる要素
        :param _path: 保存するフォルダのパス
        :param _s: size of point.
        :return: None.
        """
        colors = ListedColormap(['red', 'blue'])
        plt.xlabel(_df.columns.values[_x])
        plt.ylabel(_df.columns.values[_y])
        plt.title(_df.columns.values[_c])
        plt.scatter(_df.iloc[:, _x], _df.iloc[:, _y], c=_df.iloc[:, _c], cmap=colors, s=_s)
        plt.savefig(os.path.join(_path, 'scatter_emphasis.png'))
