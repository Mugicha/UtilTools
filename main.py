import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import UtilTools.file_operation
import pandas as pd


class App(QMainWindow):
    """
    MEMO: クラス名の後にあるカッコ内のやつは、継承するクラスを指す
    """

    def __init__(self):
        super().__init__()
        self.title = 'Util Tools for all people.'
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
        self.tabs.addTab(self.tab2, "Analyse")
        self.tab1.layout = QVBoxLayout(self)

        # For TAB1
        self.importFileButton = QPushButton("Import File.")
        self.importFileButton.clicked.connect(self.on_click_importfileBTN)
        self.tab1.layout.addWidget(self.importFileButton)
        self.tab1.setLayout(self.tab1.layout)

        # For TAB2
        self.tab2.layout = QVBoxLayout(self)  # define tab layout.

        # -Define table widget.
        self.tableWidget = QTableWidget()
        self.tableWidget.move(0, 0)

        # -Define analyse menu widget.
        # Todo: create checkbox which select column.
        self.startAnalyseButton = QPushButton("Analyse.")
        self.startAnalyseButton.clicked.connect(self.on_click_analyse)

        # Add widget to layout
        self.tab2.layout.addWidget(self.tableWidget)
        self.tab2.layout.addWidget(self.startAnalyseButton)

        self.tab2.setLayout(self.tab2.layout)
        self.layout.addWidget(self.tabs)  # Add tabs to widget
        self.setLayout(self.layout)

    @pyqtSlot()
    def on_click_importfileBTN(self):
        """
        import Excel to Dataframe and show them to Table widget on tab2.
        :return:
        """
        print('file name: ' + str(self.tab1.dragged_file))
        fope = UtilTools.file_operation.FileOperation()
        self.importDF = fope.excel_to_df(self.tab1.dragged_file)  # type: pd.DataFrame
        print('Import done.')

        # Show dataframe to table widget.
        self.tableWidget.setColumnCount(len(self.importDF.columns))
        self.tableWidget.setRowCount(len(self.importDF))
        self.tableWidget.setHorizontalHeaderLabels(self.importDF.columns)
        for idx, col in enumerate(self.importDF.columns):
            self.tableWidget.setItem(0, idx, QTableWidgetItem(str(self.importDF.iat[0, idx])))

    @pyqtSlot()
    def on_click(self):
        self.label2_1.setText('Clicked.')

    @pyqtSlot()
    def on_click_analyse(self):
        import UtilTools.qtmodule.analyse_module
        analyse = UtilTools.qtmodule.analyse_module.Analyse_module()
        analyse.dataframe_analyse(_df=self.importDF)


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
