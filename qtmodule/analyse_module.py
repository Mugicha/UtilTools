from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
import UtilTools.plot
import os
import pandas as pd


class Analyse_module:
    def dataframe_analyse(self, _df: pd.DataFrame, _x: int, _y: int, _title: str, _output_file_name: str, _output_folder_path: str):
        p = UtilTools.plot.Plot()
        p.pure_2d_scatter(_df=_df, _x=_x, _y=_y, _title=_title, _output_file_name=_output_file_name, _output_folder_path=_output_folder_path)
        print('analyse done.')
        analyse_widget = QWidget()
        analyse_widget.layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap(os.path.join(_output_folder_path, _output_file_name))
        label.setPixmap(pixmap)
        analyse_widget.layout.addWidget(label)
        analyse_widget.setLayout(analyse_widget.layout)

        return analyse_widget
