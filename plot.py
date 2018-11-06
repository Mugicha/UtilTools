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

    def pure_2d_scatter(self, _df: pd.DataFrame, _x: int, _y: int, _t='pure_2d_scatter', _s=0.5, _path='./', _f='pure_2d_scatter.png'):
        """
        DataFrame内の2列を使って二次元プロットを行う機能。
        :param _df: 使用するDataFrame
        :param _x: x軸のラベル番号
        :param _y: y軸のラベル番号
        :param _t: タイトル
        :param _s: Size of scatter plot.
        :param _path: folder path for saving figure.
        :param _f: file name of figure.
        :return: matplotlib.pyplotクラス
        """
        plt.figure(figsize=(8, 6))  # plot size
        plt.title(_t)
        plt.scatter(_df.iloc[:, _x].values, _df.iloc[:, _y].values, s=_s)
        plt.savefig(os.path.join(_path, _f))
        return plt

    def corr_map(self, _df: pd.DataFrame, _path='./'):
        """
        各変数の相関係数のヒートマップ画像を保存する機能。
        :param _df: ヒートマップの基となるDataFrame.
        :param _path: 保存するフォルダのパス
        :return: None.
        """
        plt.figure(figsize=(8, 6))  # heat map size
        sns.set(font_scale=0.6)
        sns.heatmap(_df.corr(), annot=True, cmap='plasma', linewidths=.5, annot_kws={"size": 5})
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

    def colormap(self, _df: pd.DataFrame, _x: str, _y: str, _z: str, _path='./'):
        """
        二次元カラーマップを作製する機能
        :param _df: プロットの基となるDataFrame
        :param _x: x軸としたい項目名
        :param _y: y軸としたい項目名
        :param _z: z（等高線）で示したい項目名
        :param _path: 保存するフォルダのパス
        :return: None
        """
        plt.figure(figsize=(16, 12))
        z = []
        tmp = _df.loc[:, [_x, _y, _z]]
        for i in range(tmp[_y].min(), tmp[_y].max() + 1):
            z.append(tmp[tmp[_y] == i].sort_values(_x)[_z].values)
        plt.pcolormesh(_df['date'].drop_duplicates().sort_values().values, _df['キロ程'].drop_duplicates().sort_values().values, z, cmap='hsv')
        pp = plt.colorbar(orientation='vertical')
        pp.set_label(_z)
        plt.xlabel(_x)
        plt.ylabel(_y)
        plt.tight_layout()
        plt.savefig(os.path.join(_path, 'pcolormesh.png'))
