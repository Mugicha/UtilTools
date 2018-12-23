import UtilTools.plot
import pandas as pd


class Analyse_module:
    def dataframe_analyse(self, _df: pd.DataFrame, _x: int, _y: int, _title: str):
        p = UtilTools.plot.Plot()
        p.pure_2d_scatter(_df=_df, _x=_x, _y=_y, _title=_title)
        print('analyse done.')
