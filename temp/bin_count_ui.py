from bin_count import run_mpc_file

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication

Form, Window = uic.loadUiType("MedPC_Split.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
# NEED TO FIGURE OUT print(app.get_widgets())
window.show()
app.exec_()
