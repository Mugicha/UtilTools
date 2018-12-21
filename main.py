import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from . import file_operation


class App(QMainWindow):
    """
    MEMO: クラス名の後にあるカッコ内のやつは、継承するクラスを指す
    """

    def __init__(self):
        super().__init__()
        self.title = 'Util Tools for all people.'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 300
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
        self.tab2.layout = QVBoxLayout(self)
        self.label2_1 = QLabel()  # Set label into Analyse tab
        self.label2_1.setText('None.')
        self.pushButton1 = QPushButton("PyQt5 button")  # Create button into Analyse tab
        self.pushButton1.clicked.connect(self.on_click)
        self.tab2.layout.addWidget(self.label2_1)  # Add layout.
        self.tab2.layout.addWidget(self.pushButton1)
        self.tab2.setLayout(self.tab2.layout)
        self.layout.addWidget(self.tabs)  # Add tabs to widget
        self.setLayout(self.layout)

    @pyqtSlot()
    def on_click_importfileBTN(self):
        print('file name: ' + str(DropWidget.dragged_file))
        fope = file_operation.FileOperation()
        a = DropWidget.dragged_file
        self.importDF = fope.excel_to_df(DropWidget.dragged_file)
        print('Import done.')

    @pyqtSlot()
    def on_click(self):
        self.label2_1.setText('Clicked.')


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
