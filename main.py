import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import UtilTools.file_operation
import UtilTools.common
import pandas as pd


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Util Tools for all engineer. Have a good Analysis !'
        self.left = 100
        self.top = 100
        self.width = 1000
        self.height = 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = AnalyseMasterWidget(self)
        self.setCentralWidget(self.table_widget)
        self.show()


class AnalyseMasterWidget(QWidget):
    NOT_DETECT_MSG = 'Not selected.'

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = DropWidget()
        self.tab2 = QWidget()
        self.tabs.resize(350, 250)

        # Add tabs
        self.tabs.addTab(self.tab1, "Import")
        self.tabs.addTab(self.tab2, "Row data")
        self.tab1.layout = QVBoxLayout(self)

        # For TAB1: Import
        self.importFileButton = QPushButton("Import File.")
        self.importFileButton.clicked.connect(self.on_click_importfileBTN)
        self.tab1.layout.addWidget(self.importFileButton)
        self.tab1.setLayout(self.tab1.layout)

        # For TAB2: Row data
        self.tab2.layout = QVBoxLayout(self)  # define tab layout.

        # -Define table widget.
        self.tableWidget = QTableWidget()
        self.tableWidget.move(0, 0)
        self.tableWidget.itemSelectionChanged.connect(self.itemSelectionChangedCallback)  # 選択されるセル等が変わるとcallされる

        # -Define analyse menu widget.
        self.is_column_selected = [False, False]
        self.selectedColumn1 = QLabel(self.NOT_DETECT_MSG)
        self.selectedColumn2 = QLabel(self.NOT_DETECT_MSG)
        self.startAnalyseButton = QPushButton("Analyse.")
        self.startAnalyseButton.clicked.connect(self.on_click_analyse)

        # Add widget to layout
        self.tab2.layout.addWidget(self.tableWidget)
        self.tab2.layout.addWidget(self.selectedColumn1)
        self.tab2.layout.addWidget(self.selectedColumn2)
        self.tab2.layout.addWidget(self.tableWidget)
        self.tab2.layout.addWidget(self.startAnalyseButton)

        self.tab2.setLayout(self.tab2.layout)
        self.layout.addWidget(self.tabs)  # Add tabs to widget
        self.setLayout(self.layout)

    ###########################
    # TAB: Import
    ###########################
    @pyqtSlot()
    def on_click_importfileBTN(self):
        """
        import Excel to Dataframe and show them to Table widget on tab2.
        :return:
        """
        print('file name: ' + str(self.tab1.dragged_file))
        fope = UtilTools.file_operation.FileOperation()
        self.importDF = fope.excel_to_df(self.tab1.dragged_file)  # type: pd.DataFrame
        self.columns = list(self.importDF.columns)  # type: list
        print('Import done.')

        # Show dataframe to table widget.
        self.tableWidget.setColumnCount(len(self.importDF.columns))
        self.tableWidget.setRowCount(len(self.importDF))
        self.tableWidget.setHorizontalHeaderLabels(self.importDF.columns)
        for row in range(len(self.importDF)):
            for col in range(len(self.importDF.columns)):
                self.tableWidget.setItem(row, col, QTableWidgetItem(str(self.importDF.iat[row, col])))

    ###########################
    # TAB: Row Data
    ###########################
    def itemSelectionChangedCallback(self):
        """
        QtableWidgetにて、ユーザーが選択するセルが変わるとcallされる箇所。
        現在選択されるセルをLabelで表示し、後続のグラフ描画時の軸として認識される。
        :return:
        """
        column = self.tableWidget.currentColumn()
        if self.is_column_selected[0] and self.is_column_selected[1]:
            self.is_column_selected[0] = True
            self.selectedColumn1.setText(self.importDF.columns[column])
            self.is_column_selected[1] = False
            self.selectedColumn2.setText(self.NOT_DETECT_MSG)
        elif self.is_column_selected[0] and not self.is_column_selected[1]:
            self.is_column_selected[1] = True
            self.selectedColumn2.setText(self.importDF.columns[column])
        elif not self.is_column_selected[0] and not self.is_column_selected[1]:
            self.is_column_selected[0] = True
            self.selectedColumn1.setText(self.importDF.columns[column])
            self.is_column_selected[1] = False
            self.selectedColumn2.setText(self.NOT_DETECT_MSG)
        else:
            self.is_column_selected[0] = False
            self.selectedColumn1.setText(self.NOT_DETECT_MSG)
            self.is_column_selected[1] = False
            self.selectedColumn2.setText(self.NOT_DETECT_MSG)

    @pyqtSlot()
    def on_click_analyse(self):
        """
        analyseボタンを押したときの挙動。
        新しいタブを生成し、作成したグラフを表示する。
        :return:
        """
        import UtilTools.qtmodule.analyse_module
        import UtilTools.common
        c = UtilTools.common.Common()
        if self.is_column_selected[0] and self.is_column_selected[1]:
            output_file_path = c.file_exist_check(self.selectedColumn1.text() + '_vs_' + self.selectedColumn2.text() + '.png')
            analyse = UtilTools.qtmodule.analyse_module.Analyse_module()
            self.result = analyse.dataframe_analyse(_df=self.importDF,
                                                    _x=self.columns.index(self.selectedColumn1.text()),
                                                    _y=self.columns.index(self.selectedColumn2.text()),
                                                    _title=self.selectedColumn1.text() + ' vs ' + self.selectedColumn2.text(),
                                                    _output_folder_path='./qtmodule/saved/pure_2d_scatter',
                                                    _output_file_name=output_file_path)  # type: QWidget
            self.tabs.addTab(self.result, output_file_path)
            self.setLayout(self.layout)
        else:
            QMessageBox.about(self, "Warn", "プロットしたい列を2つ選んでください。")


class DropWidget(QWidget):

    def __init__(self, parent=None):
        super(DropWidget, self).__init__(parent)
        self.setAcceptDrops(True)
        self.dragged_file = 'None.'

    def dragEnterEvent(self, event):
        event.accept()
        pass

    def dropEvent(self, event):
        event.accept()
        self.dragged_file = event.mimeData().urls()[0].toString()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # define application.
    ex = App()  # Create Widget.
    sys.exit(app.exec_())  # Exit Application.
