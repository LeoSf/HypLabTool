#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biong. Leandro D. Medus
Ph.D Student GPDD - ETSE
Universitat de València
leandro.d.medus@uv.es

07-04-2019


App Description:
----------------
Main Application to read an RGB image from a multi-spectral image and select different regions of interest when the user
clicks on the loaded image.
"""

__author__ = "Leandro D. Medus <leandro.d.medus@uv.es>"
__version__ = '0.1.0'


from ui.mainWindow import *
from src.InteraceController import InterfaceController


if __name__ == "__main__":
    import sys

    # Create the application.
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    controller = InterfaceController(ui)

    MainWindow.show()
    sys.exit(app.exec_())

