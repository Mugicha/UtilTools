import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot


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
        self.tab2.layout = QVBoxLayout(self)

        # Set label into Analyse tab
        self.label = QLabel()
        self.label.setText('None.')

        # Create button into Analyse tab
        self.pushButton1 = QPushButton("PyQt5 button")
        self.pushButton1.clicked.connect(self.on_click)

        # Add layout.
        self.tab2.layout.addWidget(self.label)
        self.tab2.layout.addWidget(self.pushButton1)
        self.tab2.setLayout(self.tab2.layout)

        # Add tabs to widget        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    @pyqtSlot()
    def on_click(self):
        print("\n")
        self.label.setText('Clicked.')
        # for currentQTableWidgetItem in self.tableWidget.selectedItems():
        #     print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


class DropWidget(QWidget):

    def __init__(self, parent=None):
        super(DropWidget, self).__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dragEnterEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        print()

    def dropEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dropEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        print()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # define application.
    ex = App()  # Create Widget.
    sys.exit(app.exec_())  # Exit Application.
