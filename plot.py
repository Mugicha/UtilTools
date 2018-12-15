import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import os
from . import common


class Plot:

    @staticmethod
    def pure_2d_plot(_df: pd.DataFrame,
                     _x: int,
                     _y: list,
                     _figsize=(16, 12),
                     _title='pure_2d_plot',
                     _output_folder_path='./',
                     _output_file_name='pure_2d_plot.png'):
        """
        DataFrame内の2列を使って二次元プロットを行う機能。
        :param _df: 使用するDataFrame
        :param _x: x軸のラベル番号
        :param _y: y軸のラベル番号
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12)
        :param _title: グラフのタイトル
        :param _output_folder_path: folder path for saving figure.
        :param _output_file_name: file name of figure.
        :return: matplotlib.pyplotクラス
        """
        plt.figure(figsize=_figsize)  # plot size
        plt.title(_title)
        plt.tight_layout()
        column = _df.columns.values
        for data in _y:
            plt.plot(_df.iloc[:, _x].values, _df.iloc[:, data].values, label=column[data])
        if len(_y) > 1:
            plt.legend()
        # plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()
        # return plt

    @staticmethod
    def pure_2d_scatter(_df: pd.DataFrame,
                        _x: int,
                        _y: int,
                        _figsize=(16, 12),
                        _title='pure_2d_scatter',
                        _s=0.5,
                        _output_folder_path='./',
                        _output_file_name='pure_2d_scatter.png'):
        """
        DataFrame内の2列を使って二次元プロットを行う機能。
        :param _df: 使用するDataFrame
        :param _x: x軸のラベル番号
        :param _y: y軸のラベル番号
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12)
        :param _title: タイトル
        :param _s: Size of scatter plot.
        :param _output_folder_path: folder path for saving figure.
        :param _output_file_name: file name of figure.
        :return: matplotlib.pyplotクラス
        """
        c = common.Common()
        plt.figure(figsize=_figsize)  # plot size
        plt.title(_title)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.scatter(_df.iloc[:, _x].values, _df.iloc[:, _y].values, s=_s)
        if c.folder_check(_output_folder_path):
            plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.close()
        # return plt

    @staticmethod
    def corr_map(_df: pd.DataFrame,
                 _output_folder_path='./',
                 _output_file_name='corr_map.png',
                 _figsize=(16, 12),
                 _graph_title='corr_map'):
        """
        各変数の相関係数のヒートマップ画像を保存する機能。
        :param _df: ヒートマップの基となるDataFrame.
        :param _output_folder_path: 保存するフォルダのパス（デフォルト：カレントディレクトリ）
        :param _output_file_name: 保存する画像ファイル名（デフォルト：corr_map.png）
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12)
        :param _graph_title: グラフのタイトル（デフォルト：corr_map）
        :return: None.
        """
        plt.figure(figsize=_figsize)  # heat map size
        # plt.title(_graph_title)
        sns.set(font_scale=0.6)
        sns.heatmap(_df.corr(), annot=True, cmap='plasma', linewidths=.5, annot_kws={"size": 5}, vmin=-1, vmax=1)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.close()

    @staticmethod
    def scatter_with_histogram(_df: pd.DataFrame,
                               dim_reduction: bool=False,
                               _figsize: tuple=(16, 12),
                               _output_folder_path: str='./',
                               _output_file_name: str='scatter_with_histogram.png'):
        """
        ヒストグラム付き散布図の画像を保存する機能。
        :param _df: 散布図の基となるDataFrame.
        :param dim_reduction: 次元削減するかどうか(default=False)
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12)
        :param _output_folder_path: 保存するフォルダのパス
        :param _output_file_name: 保存する画像ファイル名（デフォルト：scatter_with_histogram.png）
        :return: None.
        """
        plt.figure(figsize=_figsize)
        plt.xticks(rotation=90)
        if dim_reduction:
            from .data_mining import DataManipulation
            pca = DataManipulation()
            _trn = pd.DataFrame(pca.pca_reduction(_df, 2, False))
            sns.jointplot(0, 1, _trn, kind='scatter')
            plt.savefig(_output_file_name)
            return None
        plt.tight_layout()
        sns.jointplot(0, 1, _df, kind='scatter')
        plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.close()

    @staticmethod
    def pair_plot(_df: pd.DataFrame,
                  _figsize: tuple=(16, 12),
                  _output_folder_path: str='./',
                  _output_file_name: str='pair_plot.png'):
        """
        seabornのPariplot画像を保存する機能。
        :param _df: pairplotしたいDataFrame
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12))
        :param _output_folder_path: 保存するフォルダのパス
        :param _output_file_name: 保存する画像ファイル名（デフォルト：pair_plot.png）
        :return: None.
        """
        plt.figure(figsize=_figsize)
        plt.xticks(rotation=90)
        plt.tight_layout()
        sns.pairplot(_df)
        plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.close()

    @staticmethod
    def scatter_emphasis(_df: pd.DataFrame,
                         _x: int,
                         _y: int,
                         _c: int,
                         _figsize: tuple=(16, 12),
                         _output_folder_path: str='./',
                         _output_file_name: str='scatter_emphasis.png',
                         _s: int=2):
        """
        ラベルによって色を変えて散布図を表示する機能。
        :param _df: 表示させたいDataFrame
        :param _x: x軸の要素
        :param _y: y軸の要素
        :param _c: 色を変えるラベルとなる要素
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12))
        :param _output_folder_path: 保存するフォルダのパス
        :param _output_file_name: 保存する画像ファイル名（デフォルト：scatter_emphasis.png）
        :param _s: size of point.
        :return: None.
        """
        plt.figure(figsize=_figsize)
        color_box = ['red', 'm', 'darkorange', 'g', 'c', 'blue', 'darkviolet', 'brown', 'yellow', 'lime']
        number_of_label = len(common.Common().remove_duplication(_df.iloc[:, _c].values))
        if number_of_label > 10:
            print('[Warn] The number of label is too large. ' +
                  'Reduce it less than 10. Or add color to the variable named "color_box" in scatter_emphasis of plot.py.')
        color_label = color_box[0:number_of_label]
        colors = ListedColormap(color_label)
        plt.xlabel(_df.columns.values[_x])
        plt.ylabel(_df.columns.values[_y])
        plt.title(_df.columns.values[_c])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.scatter(_df.iloc[:, _x], _df.iloc[:, _y], c=_df.iloc[:, _c], cmap=colors, s=_s)
        plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.close()

    @staticmethod
    def colormap(_df: pd.DataFrame,
                 _x: str,
                 _y: str,
                 _z: str,
                 _figsize: tuple=(16, 12),
                 _output_folder_path: str='./',
                 _output_file_name: str='pcolormesh.png'):
        """
        二次元カラーマップを作製する機能
        :param _df: プロットの基となるDataFrame
        :param _x: x軸としたい項目名
        :param _y: y軸としたい項目名
        :param _z: z（等高線）で示したい項目名
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12))
        :param _output_folder_path: 保存するフォルダのパス
        :param _output_file_name: 保存する画像ファイル名（デフォルト：pcolormesh.png）
        :return: None
        """
        plt.figure(figsize=_figsize)
        z = []
        tmp = _df.loc[:, [_x, _y, _z]]
        for i in range(tmp[_y].min(), tmp[_y].max() + 1):
            z.append(tmp[tmp[_y] == i].sort_values(_x)[_z].values)
        plt.pcolormesh(_df[_x].drop_duplicates().sort_values().values, _df[_y].drop_duplicates().sort_values().values, z, cmap='hsv')
        pp = plt.colorbar(orientation='vertical')
        pp.set_label(_z)
        plt.xlabel(_x)
        plt.ylabel(_y)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(_output_folder_path, _output_file_name))
        plt.close()

    @staticmethod
    def show_correlogram(_df: pd.DataFrame,
                         _col: int,
                         _lag: int=10,
                         _figsize: tuple=(16, 12),
                         _output_folder_path: str='./',
                         _output_file_name: str='correlogram.png'):
        """
        コレログラムのグラフを画像として保存する機能。欠損値を補完しておく必要がある。
        :param _df: 対象のデータフレーム
        :param _col: コレログラムでプロットしたいデータフレームの列名
        :param _lag: コレログラムで表示するラグ(デフォルト値: 10）
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12))
        :param _output_file_name: 保存する画像ファイルの名前（デフォルト値: correlogram.png）
        :param _output_folder_path: 保存先のパス（デフォルト値: カレントディレクトリ）
        :return:
        """
        import statsmodels.api as sm
        fig, ax = plt.subplots(nrows=1, figsize=_figsize)
        plt.xlabel('Lag')
        plt.ylabel('Correlation coefficient')
        sm.graphics.tsa.plot_acf(_df.iloc[:, _col], lags=_lag, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(_output_folder_path, _output_file_name))

    def plot_all_combination(self, _df: pd.DataFrame, _cols: list, _x: int, _y: int):
        """
        DataFrameの中にあるキー情報の全組合せにてグラフをプロットする機能。
        :param _df: 対象のDataFrame
        :param _cols: キー情報としたい列名（column名で）
        :param _x: プロットするときのx軸（番号）
        :param _y: プロットするときのy軸（番号）
        :return:
        """
        import UtilTools.common
        c = UtilTools.common.Common()
        combination = c.extract_all_combination(_df, _cols)
        for each_combination in combination:
            _df_extracted = _df[_df[_cols[0] == each_combination[0]] & _df[_cols[1] == each_combination[1]]]
            self.pure_2d_scatter(_df=_df_extracted, _x=_x, _y=_y)
