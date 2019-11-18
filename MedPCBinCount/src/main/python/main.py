import os
import logging
import sys

from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog

from bin_count import run_mpc_file, exact_split_divide


def log_exception(ex, more_info=""):
    """
    Log an expection and additional info
    Parameters
    ----------
    ex : Exception
        The python exception that occured
    more_info :
        Additional string to log

    Returns
    -------
    None

    """

    template = "{0} because exception of type {1} occurred. Arguments:\n{2!r}"
    message = template.format(more_info, type(ex).__name__, ex.args)
    logging.error(message)


class DesignerUI:
    def __init__(self, design_location):
        Form, Window = uic.loadUiType(design_location)
        self.appctxt = ApplicationContext()
        self.window = Window()
        self.ui = Form()
        self.ui.setupUi(self.window)
        self.file_dialog = QFileDialog()

    def start(self):
        self.window.show()
        self.exit_code = self.appctxt.app.exec_()

    def getWidgets(self):
        return self.appctxt.app.allWidgets()

    def getWidgetNames(self):
        return [w.objectName() for w in self.getWidgets()]


class BinCountUI(DesignerUI):
    def __init__(self, design_location):
        super().__init__(design_location)
        self.selected_file = None
        self.even_split = False
        self.linkNames()
        self.setup()

    def linkNames(self):
        self.file_select_text = self.ui.FileSelect
        self.file_select_button = self.ui.FileSelectButton
        self.run_button = self.ui.pushButton
        self.splits_box = self.ui.SplitsSpinBox
        self.info_text = self.ui.lineEdit_2

    def setup(self):
        self.file_select_button.clicked.connect(
            self.selectFile)
        self.run_button.clicked.connect(
            self.run)
        self.splits_box.valueChanged.connect(
            self.onSplitsChange)
        self.splits_box.setValue(4)
        self.info_text.setText("Log messages will appear here...")

    def selectFile(self):
        self.selected_file, _filter = self.file_dialog.getOpenFileName()
        self.file_select_text.setText(self.selected_file)
        self.onSplitsChange()

    def onSplitsChange(self):
        try:
            divides, total = exact_split_divide(
                self.selected_file, self.splits_box.value())
        except Exception as e:
            log_exception(e, "During loading file")
            self.info_text.setText("Selected file could not be parsed")
            self.even_split = False
            return
        if divides:
            self.info_text.setText("Press Run to start...")
            self.even_split = True
        else:
            self.info_text.setText("{} does not divide {} trials".format(
                self.splits_box.value(), total))
            self.even_split = False

    def run(self):
        try:
            if self.selected_file is None:
                self.info_text.setText("No file selected, please select one")
                return
            if not self.even_split:
                return
            out_name = os.path.splitext(
                self.selected_file)[0] + ".csv"
            n_splits = self.splits_box.value()
            run_mpc_file(
                self.selected_file, out_name, n_splits)
            self.info_text.setText(
                "Success, file saved to {}, you can run a new file".format(
                    out_name))
        except Exception as e:
            log_exception(e, "During program execution")
            here = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(here, 'binCount.log')
            self.info_text.setText(
                "Error during execution, please check {}".format(
                    filename))


if __name__ == "__main__":
    here = os.path.dirname(os.path.realpath(__file__))
    logging.basicConfig(
        filename=os.path.join(here, 'binCount.log'),
        filemode='w', level=logging.WARNING)
    ui_location = os.path.join(here, "MedPC_Split.ui")
    ui = BinCountUI(ui_location)
    # print(ui.getWidgetNames())
    ui.start()
    sys.exit(ui.exit_code)
