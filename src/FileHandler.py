#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biong. Leandro D. Medus
Ph.D Student GPDD - ETSE
Universitat de València
leandro.d.medus@uv.es

15-04-2019

Script Description:
----------------

"""

__author__ = "Leandro D. Medus <leandro.d.medus@uv.es>"
__version__ = '0.1.0'

from src.defines import *
import os
import re


class FileHandler:
    def __init__(self):
        """
        init method of the file handler class.

        """
        self.file_path = ""
        self.points_dict = dict()

    def setFilePath(self, file_path):
        """
        Set the filepath to search images files by default
        :param file_path:   relative file path
        :return:            nothing
        """
        self.file_path = file_path

    @staticmethod
    def std_string_to_file(name, x, y, anomaly, comment, path):
        """
        Standart string format to store information in the csv file.
        :param name: name of the image without extension
        :param x: x coordinate of the anomaly
        :param y: y coordinate of the anomaly
        :param anomaly: flag to indicate if it is an anomaly (True: Anomaly)
        :param comment: comment of the anomaly detected
        :param path: path of the image
        :return: string with the standart format
        """

        return name + ", " + x + ", " + y + ", " + anomaly + ", " + comment + ", " + path + "\r\n"

    def SaveSelectedPoints(self, img_name, points, description, file_path):
        """
        TODO improve method with database use
        :param img_name:    name of the file image
        :param points:      list of points of interest
        :param description: list of descriptions of the previous points
        :param file_path:   list of paths of images

        :return:            nothing
        """
        fd = open(local_path_data_saved, "a+")

        # changing the stored name of the image in case is the default name
        regex = re.compile(r'.*/(.+)/Current Scan-RGB\.tif', re.I)
        if img_name == "Current Scan-RGB.tif":
            img_name = re.findall(regex, file_path)[0] #+ ".tif"
        else:
            img_name = re.sub(r'[\s|\-|\_]*RGB.*', '', img_name)

        # storing the current image in the file
        for i in range(len(points)):
            fd.write(self.std_string_to_file(img_name, str(points[i][0]), str(points[i][1]),
                                             str(points[i][2]), description[i], file_path))

        fd.close()

    def listAvailableImages(self):
        """
        Method to list  all the images in the directory with all the file formats (rgb and hyper-spectral files) which
        are ready to be processed.
        :return: list of available images
        :return: list of paths of each image
        """

        # get all the files in the directory (recursive)
        files_w_path = []
        for path, subdirs, files in os.walk(self.file_path):
            for name in files:
                # files_w_path.append(os.path.join(path, name))
                path = path.replace('//', '/')
                files_w_path.append(path + "/" + name)

        # checking for only files complete (RGB and datacube information
        img_hyp_list = [f for f in files_w_path if f.endswith('.bil')]      # hyper-spectral images
        img_rgb_list = [f for f in files_w_path if f.endswith('.tif')]      # rgb images

        img_hyp_list = [re.sub(r'\.bil+$', '', f) for f in img_hyp_list]      # removing extension
        img_rgb_list = [re.sub(r'[\s|\-]\w{3}\..+$', '', f) for f in img_rgb_list]      # keeping only the common part of the name

        # list of files complete
        images_complete_list = [filename for filename in img_rgb_list if filename in img_hyp_list]    # get


        # list of path for each image found
        img_paths = [f + ".bil" for f in images_complete_list]
        # regex expression to visualize better the information in the GUI
        regex_std = re.compile(r'.*/(.+)/Current Scan$', re.I)  # default name by the software
        regex_normal = re.compile(r'.*/(.+)$', re.I)            # typical name given by the user
        for i in range(len(images_complete_list)):
            if images_complete_list[i].find("Current Scan") != -1:
                images_complete_list[i] = regex_std.findall(images_complete_list[i])[0]
            else:
                images_complete_list[i] = regex_normal.findall(images_complete_list[i])[0]

        return images_complete_list, img_paths     # list of images ok

    @staticmethod
    def listSelectedPoints():
        """
        Extract selected point data from file avoiding data corruption.
        :return: list of points, where each one is [image name, x_coord, y_coord, anomaly, comment, path]
        """
        # regex = re.compile(r'(.+)\sRGB\..+,\s(\d+),\s(\d+),\s(\w+),\s(.*),\s(.*)(?!,)', re.I)
        regex = re.compile(r'(.+)[\..+]?,\s(\d+),\s(\d+),\s(\w+),\s(.*),\s(.*)(?!,)', re.I)

        data = []
        with open(local_path_data_saved) as fd:
            for line in fd:
                result = regex.findall(line)
                data.append(list(result[0])) if result else None

        return data


    def pointsDictionary(self):
        """
        To get all the selected points in a dictionary format:
            key: image name
            value: array where each row is [x, y, anomaly, comment, path]
        You can operate over each image using:
            for img_name, data in points_dict.items():
        :return: dictionary of images
        """
        self.points_dict = dict()
        for point in FileHandler.listSelectedPoints():
            if point[0] in self.points_dict:
                self.points_dict[point[0]].append(point[1:])
            else:
                self.points_dict[point[0]] = [point[1:]]

        return self.points_dict

    @staticmethod
    def listFilesRecursive():
        """
        method to list all files and directories of images to rename image files according to the directory name.
        :return:
        """

        new_img_path = "/media/Datos/Heart/Work/Valencia/pygsa/imagenes201905"

        for path, subdirs, files in os.walk(new_img_path):
            # print(subdirs)
            for name in files:
                print (os.path.join(path, name))


    def add_directory_to_db(self, anomaly):
        """"
        Method to add a complete directory to the selected points so the the complete images can be processed and a
        dataset can be generated.
        All the files are listed recursively from the path set with setFilePath()
        """
        fd = open(local_path_data_saved, "a+")

        images, paths = self.listAvailableImages()

        for i in range(len(images)):
            fd.write(self.std_string_to_file(images[i], "0", "0", anomaly,
                                             "-- complete image --", paths[i]))

        fd.close()


    def get_img_path(self, img_name):
        """
        Get the path of the image stored in the csv file.
        Because the result is a list format, the item is accesed as the first element. Also the path the 5th element in
        the sub list (index 0 and index 4).
        :param img_name:
        :return:
        """
        return self.points_dict[img_name][0][5]




if __name__ == "__main__":

    file_manager = FileHandler()


    # print(FileHandler.listSelectedPoints())
    # FileHandler.listFilesRecursive()

    # ------- testing new implementation of lis available images
    # file_manager.setFilePath(local_path_img)
    # print(file_manager.listAvailableImages())
    # print(file_manager.listSelectedPoints())
    #
    # file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201905")
    # file_manager.listAvailableImages()

    # ------- testing new implementation of dictionary
    # file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201905")
    # file_manager.pointsDictionary()

    # ------- testing method to add a complete directory to the csv file
    file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201905_malas")
    file_manager.add_directory_to_db("True")

    file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201905_buenas")
    file_manager.add_directory_to_db("False")

    # -------- old images
    file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201904/img_malas_filtradas")
    file_manager.add_directory_to_db("True")

    file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201904/buenas")
    file_manager.add_directory_to_db("False")

    file_manager.setFilePath("/media/Datos/Heart/Work/Valencia/pygsa/imagenes201904/buenas 2")
    file_manager.add_directory_to_db("False")

