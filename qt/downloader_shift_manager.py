import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from downloader_shift_controller import Ui_Form
from pathlib import Path


class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.welcome()
        self.sb.clicked.connect(self.display)
        self.tb.clicked.connect(self.close)

    def welcome(self):
        self.textBrowser.setText("Hello World! I am joker, a multiple platform programmer")

    def display(self):
        source = self.open_dir_dialog()
        print(type(source))
        self.t1.setText(source)
        target = self.t2.text()
        # 利用text Browser控件对象setText()函数设置界面显示
        self.textBrowser.append("打开成功!\n" + "源文件是: " + source + ",目标是： " + target)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        return fileName if fileName else ''

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        return files if files else ''

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)

    def open_dir_dialog(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select a Directory")
        return dir_name if dir_name else ''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())
