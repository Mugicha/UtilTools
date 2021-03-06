import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from . import common
from scipy import fftpack


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
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

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
        if os.path.exists(_output_folder_path):
            plt.savefig(c.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    @staticmethod
    def simple_histogram(x: list,
                         color: list,
                         xlabel: str = 'xlabel',
                         _output_file_name: str = 'histogram.png',
                         bins: int = 10,
                         label: list = None,
                         hist_type: str = 'side-by-side',
                         ):
        """
        ヒストグラムを作成する機能。グラフはbins毎に各ラベル纏めてプロットする。

        :param x: ヒストグラムを作成したいデータ。複数ラベル分作成したい場合は、list型にまとめること。
        :param color: グラフの色。複数ラベルある場合は、複数ラベル分纏めてlist型で指定すること。
        :param xlabel: x軸の名称（default: xlabel）
        :param _output_file_name: 出力ファイル名（default: ./histogram.png）
        :param bins: ヒストグラムの分割数（default: 10）
        :param label: legendで指定する名前。複数ラベル分ある時は、その分list型にまとめて指定する事（default: None）
        :param hist_type: ヒストグラムの描画方法（[default]side-by-side: 各要素横並び、stacked: 各要素を積み上げる）
        :return:
        """
        if len(x) == 0: return
        if len(color) == 0: return
        if len(x) != len(color):
            assert '[simple_histogram] the size of each list x, colo must be same.'
        plt.ylabel('count')
        plt.xlabel(xlabel)
        stacked = False if hist_type == 'side-by-side' else True
        if label is None:
            plt.hist(x, color=color, bins=bins, stacked=stacked)
        else:
            if len(x) != len(label):
                assert '[simple_histogram] the size of each list x, label must be same.'
                return
            plt.hist(x, color=color, bins=bins, label=label, stacked=stacked)
            plt.legend()
        plt.savefig(_output_file_name)
        plt.close()

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
        plt.title(_graph_title)
        sns.set(font_scale=0.6)
        sns.heatmap(_df.corr(), annot=True, cmap='plasma', linewidths=.5, annot_kws={"size": 5}, vmin=-1, vmax=1)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    @staticmethod
    def scatter_with_histogram(_df: pd.DataFrame,
                               dim_reduction: bool = False,
                               _figsize: tuple = (16, 12),
                               _output_folder_path: str = './',
                               _output_file_name: str = 'scatter_with_histogram.png',
                               ):
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
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    @staticmethod
    def pair_plot(_df: pd.DataFrame,
                  _figsize: tuple = (16, 12),
                  _output_folder_path: str = './',
                  _output_file_name: str = 'pair_plot.png'):
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
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    @staticmethod
    def scatter_emphasis(_df: pd.DataFrame,
                         _x: int,
                         _y: int,
                         _c: int,
                         _labeldict: dict,
                         _color_box: list = None,
                         _figsize: tuple = (16, 12),
                         _output_folder_path: str = './',
                         _output_file_name: str = 'scatter_emphasis.png',
                         _s: int = 2,
                         ):
        """
        ラベルによって色を変えて散布図を表示する機能。
        :param _df: 表示させたいDataFrame
        :param _x: x軸の要素
        :param _y: y軸の要素
        :param _c: 色を変えるラベルとなる要素
        :param _labeldict: legendに表示するラベル
        :param _color_box: a list of color associated with _c.
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12))
        :param _output_folder_path: 保存するフォルダのパス
        :param _output_file_name: 保存する画像ファイル名（デフォルト：scatter_emphasis.png）
        :param _s: size of point.
        :return: None.
        """
        # 色の補色関係に従い、順に並べること
        if _color_box is None:
            color_box = ['red', 'green', 'mediumorchid', 'gold', 'blue', 'darkorange', 'cyan', 'purple', 'lime', 'black', 'darkred']
        else:
            color_box = _color_box

        # cdict と label を合わせる
        label = common.Common().remove_duplication(_df.iloc[:, _c].values)
        if len(label) > len(color_box):
            print('[Warn] The number of label is too large. ' +
                  'Reduce it less than ' + str(len(color_box)) +
                  '. Or add color to the variable named "color_box" in scatter_emphasis of plot.py.')
        cdict = {}
        for idx, c in enumerate(label):
            cdict[c] = color_box[idx]
        plt.figure(figsize=_figsize)
        fig, ax = plt.subplots()
        for g in np.unique(_df.iloc[:, _c]):
            ix = np.where(_df.iloc[:, _c] == g)
            ax.scatter(_df.iloc[:, _x].values[ix], _df.iloc[:, _y].values[ix], c=cdict[g], label=_labeldict[g], s=_s)
        ax.legend()
        plt.xticks(rotation=90)
        plt.plot(_df.iloc[:, _x].values, _df.iloc[:, 7].values)
        plt.tight_layout()
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    @staticmethod
    def colormap(_df: pd.DataFrame,
                 _x: str,
                 _y: str,
                 _z: str,
                 _figsize: tuple = (16, 12),
                 _output_folder_path: str = './',
                 _output_file_name: str = 'pcolormesh.png'):
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
        for i in range(int(tmp[_y].min()), int(tmp[_y].max() + 1)):
            z.append(tmp[tmp[_y] == i].sort_values(_x)[_z].values)
        plt.pcolormesh(_df[_x].drop_duplicates().sort_values().values, _df[_y].drop_duplicates().sort_values().values, z, cmap='hsv')
        pp = plt.colorbar(orientation='vertical')
        pp.set_label(_z)
        plt.xlabel(_x)
        plt.ylabel(_y)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    @staticmethod
    def show_correlogram(_df: pd.DataFrame,
                         _col: int,
                         _lag: int = 10,
                         _figsize: tuple = (16, 12),
                         _output_folder_path: str = './',
                         _output_file_name: str = 'correlogram.png'):
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
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()

    def simple_fft(self, _series: pd.Series,
                   _output_file_name: str,
                   _output_folder_path: str,
                   _title: str,
                   _figsize: tuple = (16, 12)):
        """
        入力値を離散フーリエ変換し、周波数成分でプロットする機能。
        :param _series: Series to be analysed.
        :param _output_file_name:
        :param _output_folder_path:
        :param _title:
        :param _figsize: 保存する画像のサイズ（デフォルト：(16, 12))
        :return: None
        """
        data = _series.values  # type: np.ndarray
        sample_freq = fftpack.fftfreq(data[:].size, d=1)
        y_fft = fftpack.fft(data[:])
        pidxs = np.where(sample_freq > 0)
        freqs, power = sample_freq[pidxs], np.abs(y_fft)[pidxs]
        plt.figure(figsize=_figsize)
        plt.title(_title)
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Power")
        plt.plot(freqs, power, 'b-')
        plt.tight_layout()
        plt.savefig(common.Common.file_exist_check(os.path.join(_output_folder_path, _output_file_name)))
        plt.close()
