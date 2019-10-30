#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biong. Leandro D. Medus
Ph.D Student GPDD - ETSE
Universitat de València
leandro.d.medus@uv.es

12-04-2019

"""
__author__ = "Leandro D. Medus <leandro.d.medus@uv.es>"
__version__ = '0.1.0'

# importing the graphical user interface
from ui.mainWindow import *

from PyQt5 import QtCore, QtGui, QtWidgets
from src.FileHandler import FileHandler
from src.defines import *

from PyQt5.QtGui import QPixmap


class InterfaceController:
    """
    Class to control the interface. Signals and Slot are managed here.
    """
    def __init__(self, interface):
        """
        Init method - default parameters initialization of the interface
        :param interface: interface to control (mainWindow)
        """
        self.sel_points = []                # list to save coordinates and anomalies
        self.sel_points_description = []    # description of each point
        self.interface = interface
        # print(type(interface))

        # Default values for ROI
        self.interface.in_roi_height.setValue(DEFAULT_ROI_DIMENSION_H)
        self.interface.in_roi_width.setValue(DEFAULT_ROI_DIMENSION_W)

        self.connect_signals_to_methods()

        self.last_item_active = 0
        self.deleted_item = False
        self.file_name = ""
        self.file_path = ""
        self.path_images = ""

        self.items = []
        self.paths = []
        self.points_dict = dict()

        self.flag_auto_refresh = True
        self.flag_expand_files = False
        self.flag_expand_points = True

        self.file_manager = FileHandler()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.listAvailableImages)

        self.interface.treeFiles.setColumnWidth(0, 200)
        self.interface.treeSelPoints.setColumnWidth(0,200)
        self.interface.treeSelPoints.setColumnWidth(1, 300)

        self.interface.wgtDisplayArea.setDefaultDirectoryPath(local_path_img)

        # self.pix_uv = QPixmap(local_path_app_img + "logo_uv.jpg")
        # self.interface.lb_img_gpdd.setPixmap(self.pix_uv.scaled(100,100,QtCore.Qt.KeepAspectRatio))

        self.interface.actionClose.triggered.connect(QtWidgets.qApp.quit)

    def connect_signals_to_methods(self):
        """
        Standard method to connect signals and slots of the user interface
        :return: None
        """
        # Signal and Slots for display Area of the Image
        self.interface.wgtDisplayArea.leftMouseButtonPressed.connect(self.handleLeftClickImage)

        # Signal and Slots for buttons
        self.interface.btnLoadImage.clicked.connect(self.loadNewImage)
        self.interface.btnSaveImages.clicked.connect(self.saveSelectedPoints)
        self.interface.btn_rst_points.clicked.connect(self.clearSelectedPoints)
        self.interface.btn_refresh.clicked.connect(self.listAvailableImages)
        self.interface.btn_sel_dir.clicked.connect(self.select_img_directory)

        self.interface.btn_autorefresh.clicked.connect(self.enable_autorefresh)
        self.interface.btn_expand_img.clicked.connect(self.expand_files)
        self.interface.btn_expand_points.clicked.connect(self.expand_points)

        # Signal and Slots for Selected Points in QListWidget in preview tab
        self.interface.lst_selected_points.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.interface.lst_selected_points.customContextMenuRequested.connect(self.pop_menu)
        self.interface.lst_selected_points.itemSelectionChanged.connect(self.itemActive)
        # self.interface.lst_selected_points.itemActivated.connect(self.itemDoubleClick)

        # Signal and Slots for Selected elements in QTreeWidget in Pre-Processing tab
        self.interface.treeFiles.itemClicked.connect(self.updateSelectedPath)
        self.interface.treeSelPoints.itemClicked.connect(self.updateSelectedPathFromPoint)

    def loadNewImage(self):
        """
        Method to load an image in the pre-visualization area.
        :return: None
        """
        self.file_path = self.interface.wgtDisplayArea.loadImageFromFile()

        self.file_name = self.file_path[self.file_path.rfind('/')+1:]
        self.interface.lbl_image_name.setText(self.file_name[:-4])
        self.interface.lbl_image_path.setText(self.file_path)
        self.clearSelectedPoints()

        self.interface.lbl_msg_areas_saved.setText("")

    def saveSelectedPoints(self):
        """
        Method to save all the current selected points in the database.
        :return: None
        """

        if len(self.sel_points_description) > 0:
            self.sel_points_description[self.interface.lst_selected_points.currentRow()] = self.interface.box_point_description.toPlainText()

        self.file_manager.SaveSelectedPoints(self.file_name, self.sel_points,
                                             self.sel_points_description, self.file_path)

        self.interface.lbl_msg_areas_saved.setText(msg_areas_saved)

    def clearSelectedPoints(self):
        """
        Method to clear selected points from the preview windows. IMPORTANT: this method doesn't delete information from the database.
        :return: None
        """
        self.sel_points = []
        self.sel_points_description = []
        self.last_item_active = 0
        self.interface.box_point_description.setPlainText("")
        self.deleted_item = True
        self.interface.lst_selected_points.clear()

    def handleLeftClickImage(self, x, y):
        """
        Slot to handle left click in the pre-visualization area.
        Ctrl + click: add a point
        click + hold: pan over the image when zoom is active
        :param x: x coordinate given by the event
        :param y: y coordinate given by the event
        :return: None
        """
        if QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            # adding in raw data format the selected point
            self.sel_points.append([int(x), int(y), self.interface.chb_anomaly.isChecked()])
            self.sel_points_description.append("")

            # generating text to display in the list of the interface
            point_to_list = "(" + str(int(x)) + ", " + str(int(y)) + ") - "
            if self.interface.chb_anomaly.isChecked():
                point_to_list += "Anomaly"
            else:
                point_to_list += "Normal"
            self.interface.lst_selected_points.addItem(point_to_list)

    def getListOfPoints(self):
        """
        Getter of the list of selected points.
        :return:
        """
        return self.sel_points

    def itemActive(self):
        """
        Slot to handle the activation of an item from the list of selected point widget.
        This method display the information of the selected saved points during the current session. Also handle the mini-display preview based on the selected item.
        :return: None
        """
        list_size = len(self.sel_points_description)
        if not self.deleted_item and list_size > 0:
            print("active  last_item_active: " + str(self.last_item_active))
            self.sel_points_description[self.last_item_active] = self.interface.box_point_description.toPlainText()
        else:
            self.deleted_item = False

        index_wdgt = self.interface.lst_selected_points.currentRow()    # index goes from 0 to n-1
        # print("Current index: " + str(index_wdgt) + " -- value: [" + \
        #       ', '.join(str(e) for e in self.sel_points[index_wdgt]) + ']')

        # operations when there are items in the list
        if (index_wdgt >= 0) and (list_size>0) and not(index_wdgt > list_size-1):

            # update
            self.interface.box_point_description.setPlainText(self.sel_points_description[index_wdgt])
            self.last_item_active = index_wdgt

            # displaying the pre-visualization of the current active item
            self.interface.wgtDisplayPreview.setImage(
                self.interface.wgtDisplayArea.cropImage(
                    self.sel_points[index_wdgt][0], self.sel_points[index_wdgt][1],
                    # int(self.interface.in_roi_width.toPlainText()), int(self.interface.in_roi_height.toPlainText())))
                    self.interface.in_roi_width.value(), self.interface.in_roi_height.value()))
        else:
            self.last_item_active = 0

        if list_size == 0:
            self.interface.wgtDisplayPreview.clearImage()

        # TODO improve bugfix: when one item remain after a delete operation, the image is not updated
        if list_size == 1:
            self.interface.wgtDisplayPreview.setImage(
                self.interface.wgtDisplayArea.cropImage(
                    self.sel_points[0][0], self.sel_points[0][1],
                    self.interface.in_roi_width.value(), self.interface.in_roi_height.value()))


    def itemDoubleClick(self, item):
        """
        Double click slot for each element in QtWidgets of selected points in the preview window.
        :param item:
        :return:
        """
        # print(item.text())
        # print(self.interface.lst_selected_points.currentIndex())
        pass
        # TODO: TBD

    def pop_menu(self, pos):
        """
        Slot of the selected points list to handle the right click over an item.
        Pop menu for each element in QtWidgets of selected points in the preview window, showing options: delete, set anomaly, set normal.
        :param pos: ---
        :return: None
        """
        menu = QtWidgets.QMenu()
        action_delete = menu.addAction('Delete')
        action_set_anomaly = menu.addAction('Set: Anomaly')
        action_set_normal = menu.addAction('Set: Normal')
        action = menu.exec_(self.interface.lst_selected_points.mapToGlobal(pos))

        index_wdgt = self.interface.lst_selected_points.currentRow()
        # print("pop index_wdgt: " + str(index_wdgt))

        if action == action_delete:
            self.deleted_item = True

            if index_wdgt > 0:
                self.interface.box_point_description.setPlainText(self.sel_points_description[index_wdgt])
            else:
                self.interface.box_point_description.setPlainText("")

            del (self.sel_points[index_wdgt])
            del (self.sel_points_description[index_wdgt])
            self.interface.lst_selected_points.takeItem(index_wdgt)
            self.itemActive()       # function call to update the preview window

        elif action == action_set_anomaly:
            item = self.interface.lst_selected_points.currentItem()
            item_info = item.text().split(' - ')
            item.setText(item_info[0] + " - Anomaly")
            self.sel_points[index_wdgt][2] = True

        elif action == action_set_normal:
            item = self.interface.lst_selected_points.currentItem()
            item_info = item.text().split(' - ')
            item.setText(item_info[0] + " - Normal")
            self.sel_points[index_wdgt][2] = False

    def listAvailableImages(self):
        """
        Method to show all the available images in the selected directory to process and the preprocessed images in
        another view. This include images recursively in the directory.
        The list of images came from of listSelectedPoints() of FileManager.
        :return: None
        """

        # cleaning both views for available images and preprocessed images
        self.interface.treeFiles.clear()
        self.interface.treeSelPoints.clear()

        # list available images in the directory
        self.items, self.paths = self.file_manager.listAvailableImages()

        # list all the selected points from file
        selected_points = self.file_manager.listSelectedPoints()

        # Dictionary of all selected points
        self.points_dict = self.file_manager.pointsDictionary()

        # ----- treeFiles: organizing the data -----
        # Adding all the available images in the directory
        for filename in self.items:

            # level 1 item in treeFiles
            item_l1 = QtWidgets.QTreeWidgetItem([filename, "Pending..."])

            for point_data in selected_points:
                if point_data[0] == filename:
                    child_description = "(" + str(point_data[1]) + ", " + str(point_data[2]) + ")\t- "+ point_data[3]
                    item_l1.addChild(QtWidgets.QTreeWidgetItem([child_description]))
                    item_l1.setText(1, "Done!")

            self.interface.treeFiles.addTopLevelItem(item_l1)

        # ----- treeSelPoints: organizing the data -----
        for img_name, data in self.points_dict.items():

            # level 1 item in treeSelPoints
            item_l1 = QtWidgets.QTreeWidgetItem([img_name])
            for point_data in data:
                child_description = "(" + str(point_data[0]) + ", " + str(point_data[1]) + ")\t- "+ point_data[2]
                item_l1.addChild(QtWidgets.QTreeWidgetItem([child_description, point_data[3]]))

            self.interface.treeSelPoints.addTopLevelItem(item_l1)

        # Select view mode for treeSelPoints: Expanded
        self.interface.treeSelPoints.expandToDepth(0)

    def select_img_directory(self):
        """
        To select the dictory where to look all the images for the pre-processing window. Also, it start the timer for
        the refresh function of QTreeWidgetItems
        :return: None
        """
        self.path_images = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select where images are located",
            local_path_img,
            QtWidgets.QFileDialog.ShowDirsOnly
        )

        self.interface.wgtDisplayArea.setDefaultDirectoryPath(self.path_images)
        self.file_manager.setFilePath(self.path_images)
        self.interface.lbl_dir_path.setText(self.path_images)
        self.timer.start(1000)

    def updateSelectedPath(self):
        """

        :return:
        """
        item_name = self.interface.treeFiles.currentItem().text(0)

        try:
            path = self.paths[self.items.index(item_name)]
            self.interface.lbl_item_path.setText(path)
        except ValueError:
            pass

    def updateSelectedPathFromPoint(self):
        """

        :return:
        """
        item_name = self.interface.treeSelPoints.currentItem().text(0)

        try:
            values = self.points_dict[item_name]
            self.interface.lbl_item_path.setText(values[0][4])

        except KeyError or ValueError:
            pass

    def enable_autorefresh(self):
        """

        :return:
        """
        if self.flag_auto_refresh is True:
            self.flag_auto_refresh = False
            self.timer.stop()
            self.interface.btn_autorefresh.setText("AutoRefesh OFF")
        else:
            self.flag_auto_refresh = True
            self.timer.start(1000)
            self.interface.btn_autorefresh.setText("AutoRefesh ON")

    def expand_files(self):
        """

        :return:
        """
        if self.flag_expand_files is True:
            self.flag_expand_files = False
            self.interface.treeFiles.collapseAll()
            self.interface.btn_expand_img.setText("Expand Files")
        else:
            self.flag_expand_files = True
            self.interface.treeFiles.expandToDepth(0)
            self.interface.btn_expand_img.setText("Collapse Files")

    def expand_points(self):
        """

        :return:
        """
        if self.flag_expand_points is True:
            self.flag_expand_points = False
            self.interface.treeSelPoints.collapseAll()
            self.interface.btn_expand_points.setText("Expand Points")
        else:
            self.flag_expand_points = True
            self.interface.treeSelPoints.expandToDepth(0)
            self.interface.btn_expand_points.setText("Collapse Points")