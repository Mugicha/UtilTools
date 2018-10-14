import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Plot():

    def corr_map(self, _df: pd.DataFrame, path = './'):
        plt.figure(figsize=(8, 6))  # heat map size
        sns.heatmap(_df.corr(), annot=True, cmap='plasma', linewidths=.5)
        plt.savefig('corr_map.png')

    def scatter_with_histogram(self, _df: pd.DataFrame, dim_reduction: False):
        plt.figure(figsize=(8, 6))
        if dim_reduction:
            from UtilTools.data_mining import DataManipulation
            pca = DataManipulation()
            _trn = pca.pca_reduction(_df, 2, False)
            sns.jointplot(_trn.iloc[:, 0], _trn.iloc[:, 1], _df, kind='scatter')
            plt.savefig('scatter_with_histogram_pca.png')
        sns.jointplot(_df.iloc[:, 0], _df.iloc[:, 1], _df, kind='scatter')
        plt.savefig('scatter_with_histogram.png')
