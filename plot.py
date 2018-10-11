import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Plot():

    def corr_map(self, _df: pd.DataFrame):
        plt.figure(figsize=(8, 6))  # heat map size
        sns.heatmap(_df.corr(), annot=True, cmap='plasma', linewidths=.5)
        plt.savefig('output.png')
