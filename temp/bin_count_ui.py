import os
from bin_count import run_mpc_file

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication


class DesignerUI:
    def __init__(self, design_location):
        Form, Window = uic.loadUiType(design_location)
        self.app = QApplication([])
        self.window = Window()
        self.ui = Form()
        self.ui.setupUi(self.window)

    def start(self):
        self.window.show()
        self.app.exec_()

    def get_widgets(self):
        return self.app.allWidgets()

    def get_widget_names(self):
        return [w.objectName() for w in self.get_widgets()]


class BinCountUI(DesignerUI):
    def selectFile(self):
        print(self.ui.FileSelectButton)


if __name__ == "__main__":
    here = os.path.dirname(os.path.realpath(__file__))
    ui_location = os.path.join(
        here, "MedPC_Split.ui")
    ui = BinCountUI(ui_location)
    ui.start()
