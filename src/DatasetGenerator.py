#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biong. Leandro D. Medus
Ph.D Student GPDD - ETSE
Universitat de València
leandro.d.medus@uv.es

09-04-2019

Script Description:
----------------

"""

__author__ = "Leandro D. Medus <leandro.d.medus@uv.es>"
__version__ = '0.1.0'

import h5py
import re
import numpy as np
from osgeo import gdal
from src.defines import *
from src.FileHandler import FileHandler
from src.utils import *

from src.test import get_expanded_training_set


class DatasetGenerator:
    """
    Class to generate datasets from selected points saved in disk. Data is stored as a multi-dimensional matrix in HD5 format.

    TODO: integrate in the GUI
    """

    def __init__(self):
        """
        Initialization method of the class.
        """
        self.out_file_name = None
        self.roi_w = 0
        self.roi_h = 0

    def set_out_file(self, name):
        """
        Setter of the output file name.
        :param name: string of the filename.
        :return: None
        """
        self.out_file_name = name

    def set_roi(self, roi_w, roi_h):
        """

        :param roi_w:
        :param roi_h:
        :return: None
        """
        self.roi_w = roi_w
        self.roi_h = roi_h

    def getData(self, roi_w, roi_h):
        """
        Method to generate the current dataset based on the selected points saved in the csv file. If any of the
        parameters roi_w or roi_h is 0, the dataset will be generated using the complete image
        :param roi_w: region of interest width
        :param roi_h: region of interest height
        :return: None
        """
        file_manager = FileHandler()
        files_and_points = file_manager.pointsDictionary()  # complete list of images and points

        first_img_name = next(iter(files_and_points.keys()))  # first image name

        # getting the current parameters of the first image in the dictionary red from the path stored.
        # lines, samples, bands = get_parameters(files_and_points[first_img_name][0][4])
        lines, samples, bands = get_parameters(file_manager.get_img_path(first_img_name))

        cnt_imgs = -1

        if (roi_w is 0) or (roi_h is 0):
            full_image = True
            n_full_imgs = len(files_and_points)
            dataset = np.zeros([n_full_imgs, lines, samples, bands])
            dataset_ann = np.zeros([n_full_imgs, 1])
            printProgressBar(0, n_full_imgs, prefix='Progress:', suffix='Complete', length=50)
        else:
            full_image = False
            n_points = len(FileHandler.listSelectedPoints())
            dataset = np.zeros([n_points, roi_w, roi_h, bands])
            dataset_ann = np.zeros([n_points, 1])

        # iteration over each image which has associated N selected points in data
        for img_name, data_points in files_and_points.items():
            # print(img_name, "    ", data)

            # checking specific parameters of the current image
            params = get_parameters(file_manager.get_img_path(img_name))
            if params is not None:
                # retrieving data from parameters
                lines, samples, bands = params

                img_layers = np.zeros([lines, samples, bands])

                # reading all the wavelengths
                for i in range(0, bands):
                    file_path_complete = file_manager.get_img_path(img_name)  # with datacube extension
                    img_layers[:, :, i] = ReadBilFile(file_path_complete, i + 1)

                if full_image:
                    cnt_imgs = cnt_imgs + 1
                    dataset[cnt_imgs] = img_layers
                    for point in data_points:
                        # checking if there is at least one ROI with "Anomaly" annotation
                        dataset_ann[cnt_imgs] = True if point[2] == p_anomaly else None

                else:
                    for point in data_points:
                        cnt_imgs = cnt_imgs + 1
                        # lines : number of row of the image
                        # samples: number of columns of the image
                        x1, y1, x2, y2 = DatasetGenerator.check_limits(int(point[0]), int(point[1]),
                                                                       roi_w, roi_h, samples, lines)
                        dataset[cnt_imgs] = img_layers[y1:y2, x1:x2, :]
                        dataset_ann[cnt_imgs] = True if point[2] == p_anomaly else False

            # Progress report
            if full_image:
                printProgressBar(cnt_imgs + 1, n_full_imgs, prefix='Progress:', suffix='Complete', length=50)
                # print(cnt_imgs)

        # print(dataset.shape)
        # print(dataset.max())

        with h5py.File(local_path_out + 'datacube_dataset.hdf5', 'w') as f:
            # datacube_images optimized for 2 bytes integer
            f.create_dataset('datacube_images', data=dataset, dtype='i2', compression="gzip", compression_opts=9)
            f.create_dataset('datacube_annotations', data=dataset_ann, dtype='i1', compression="gzip",
                             compression_opts=9)

            # TODO raise an exception if there is any problem with the file or the number of bands read

        # return img_layers

    def sort_data_generate_dataset(self, mini_batch):
        """
        Method to generate a dataset from the selected point list saved in the csv file. The current implementation
        works with complete images and not with specifics ROIs (regions of interest)
        :param mini_batch: mi batch size
        :return:
        """
        logger = open(local_path_dataset_log, "w")
        img_data = FileHandler.listSelectedPoints()

        np.random.shuffle(img_data)
        # print(img_data)

        n_images = len(img_data)
        cnt_img = -1

        n_train = int(n_images * 0.7)  # total of images for the training set
        n_test = n_images - n_train  # total of images for the test set

        print(n_images)
        print(n_train)
        print(n_test)

        lines, samples, bands = get_parameters(img_data[0][5])

        # for training -------------------------------------------------------------------------------------------------
        n_batches = int(np.ceil(n_train / mini_batch))
        print("cant de batches for training: ", n_batches)

        # generating the first N mini batches -1 for TRAINING
        for mini_batch_index in range(int(np.floor(n_train / mini_batch))):   #original
        # for mini_batch_index in [0]:
            # print("mini-batch index test 1st part: ", mini_batch_index)

            filename = "pygsa_db_train_old_" + str(mini_batch_index)
            logger.write("*** train minibatch: " + filename + '\n')

            dataset = np.zeros([mini_batch, lines-40, samples, bands])
            dataset_ann = np.zeros([mini_batch, 1])
            # print(filename)

            # reading each image of the mini batch
            for index in range(mini_batch):
                cnt_img = cnt_img + 1
                logger.write("  [" + str(index) + "]" + str(img_data[cnt_img]) + '\n')

                img_layers = np.zeros([lines-40, samples, bands])

                # reading all the wavelengths
                for band in range(0, bands):
                    data_read = ReadBilFile(img_data[cnt_img][5], band + 1)
                    img_layers[:, :, band] = data_read[40:, :]

                dataset[index] = img_layers
                # var = img_data[cnt_img][3]
                # print("------------------ ")
                # print(type(var))
                # print(var)
                # print("------------------ ")
                dataset_ann[index] = 1 if img_data[cnt_img][3] == p_anomaly else 0

                printProgressBar(index + 1, mini_batch,
                                 prefix='Batch [' + str(mini_batch_index + 1) + '/' + str(n_batches) + '] - Progress:',
                                 suffix='Complete', length=50)

            logger.write("--- Complete dataset annotations: " + '\n')
            logger.write(str(dataset_ann) + '\n')

            print(local_path_out + filename + '.hdf5' + '\n')
            with h5py.File(local_path_out + filename + '.hdf5', 'w') as f:
                # optimized for 2 bytes integer
                f.create_dataset('datacube_images', data=dataset, dtype='i2', compression="gzip", compression_opts=9)
                f.create_dataset('datacube_annotations', data=dataset_ann, dtype='i1', compression="gzip",
                                 compression_opts=9)

        # generating the last training mini-batch
        filename = "pygsa_db_train_old_" + str(n_batches - 1)
        n_last_mini_batch_imgs = n_train % mini_batch
        dataset = np.zeros(
            [n_last_mini_batch_imgs, lines-40, samples, bands])  # cleaning the data for the current mini-batch
        dataset_ann = np.zeros([n_last_mini_batch_imgs, 1])  # cleaning the data for the current mini-batch

        for index in range(n_last_mini_batch_imgs):
            cnt_img = cnt_img + 1
            img_layers = np.zeros([lines-40, samples, bands])  # cleaning the data for the current image

            # reading all the wavelengths
            for band in range(0, bands):
                data_read = ReadBilFile(img_data[cnt_img][5], band + 1)
                img_layers[:, :, band] = data_read[40:, :]
                # img_layers[:, :, band] = ReadBilFile(img_data[cnt_img][5], band + 1)

            dataset[index] = img_layers
            dataset_ann[index] = 1 if img_data[cnt_img][3] == p_anomaly else 0

            printProgressBar(index + 1, n_last_mini_batch_imgs,
                             prefix='Batch [' + str(n_batches) + '/' + str(n_batches) + '] - Progress:',
                             suffix='Complete', length=50)

        print(local_path_out + filename + '.hdf5' + '\n')
        with h5py.File(local_path_out + filename + '.hdf5', 'w') as f:
            # optimized for 2 bytes integer
            f.create_dataset('datacube_images', data=dataset, dtype='i2', compression="gzip", compression_opts=9)
            f.create_dataset('datacube_annotations', data=dataset_ann, compression="gzip", compression_opts=9)

        # for testing  -------------------------------------------------------------------------------------------------
        n_batches = int(np.ceil(n_test / mini_batch))
        print("cant de batches for testing: ", n_batches)

        # generating the first N mini batches -1 for TEST
        for mini_batch_index in range(int(np.floor(n_test / mini_batch))):
            # print("mini-batch index test 1st part: ", mini_batch_index)

            filename = "pygsa_db_test_old_" + str(mini_batch_index)
            dataset = np.zeros([mini_batch, lines-40, samples, bands])
            dataset_ann = np.zeros([mini_batch, 1])
            # print(filename)

            # reading each image of the mini batch
            for index in range(mini_batch):
                cnt_img = cnt_img + 1
                img_layers = np.zeros([lines-40, samples, bands])

                # reading all the wavelengths
                for band in range(0, bands):
                    data_read = ReadBilFile(img_data[cnt_img][5], band + 1)
                    img_layers[:, :, band] = data_read[40:, :]
                    # img_layers[:, :, band] = ReadBilFile(img_data[cnt_img][5], band + 1)

                dataset[index] = img_layers
                dataset_ann[index] = 1 if img_data[cnt_img][3] == p_anomaly else 0

                printProgressBar(index + 1, mini_batch,
                                 prefix='Test Batch [' + str(mini_batch_index + 1) + '/' + str(
                                     n_batches) + '] - Progress:',
                                 suffix='Complete', length=50)

            print(local_path_out + filename + '.hdf5' + '\n')
            with h5py.File(local_path_out + filename + '.hdf5', 'w') as f:
                # optimized for 2 bytes integer
                f.create_dataset('datacube_images', data=dataset, dtype='i2', compression="gzip", compression_opts=9)
                f.create_dataset('datacube_annotations', data=dataset_ann, compression="gzip", compression_opts=9)

        # reading the last training mini-batch
        filename = "pygsa_db_test_old" + str(n_batches - 1)
        n_last_mini_batch_imgs = n_test % mini_batch
        dataset = np.zeros(
            [n_last_mini_batch_imgs, lines-40, samples, bands])  # cleaning the data for the current mini-batch
        dataset_ann = np.zeros([n_last_mini_batch_imgs, 1])  # cleaning the data for the current mini-batch

        for index in range(n_last_mini_batch_imgs):
            cnt_img = cnt_img + 1
            img_layers = np.zeros([lines-40, samples, bands])  # cleaning the data for the current image

            # reading all the wavelengths
            for band in range(0, bands):
                data_read = ReadBilFile(img_data[cnt_img][5], band + 1)
                img_layers[:, :, band] = data_read[40:, :]
                # img_layers[:, :, band] = ReadBilFile(img_data[cnt_img][5], band + 1)

            dataset[index] = img_layers
            dataset_ann[index] = 1 if img_data[cnt_img][3] == p_anomaly else 0

            printProgressBar(index + 1, n_last_mini_batch_imgs,
                             prefix='Test Batch [' + str(n_batches) + '/' + str(n_batches) + '] - Progress:',
                             suffix='Complete', length=50)

        print(local_path_out + filename + '.hdf5' + '\n')
        with h5py.File(local_path_out + filename + '.hdf5', 'w') as f:
            # optimized for 2 bytes integer
            f.create_dataset('datacube_images', data=dataset, dtype='i2', compression="gzip", compression_opts=9)
            f.create_dataset('datacube_annotations', data=dataset_ann, compression="gzip", compression_opts=9)

        print("cnt_img: ", cnt_img)
        # print(n_images)
        # print(n_test)
        # print(n_batches)

    @staticmethod
    def allocate_mem_dataset(mini_batch_size, dimensions):
        """
        TODO english description
        :param mini_batch_size:
        :param dimensions:
        :return: empty_dataset
        """
        if dimensions is None:
            empty_dataset = np.zeros([mini_batch_size] + DEFAULT_DIMENSIONS)
        else:
            empty_dataset = np.zeros([mini_batch_size] + dimensions)

        return empty_dataset

    @staticmethod
    def write_dataset_to_file(filename, dataset, dataset_ann, imgs_metadata):
        """
        TODO english description

        :param filename: complete file name of the image
        :param dataset: multi-dimensional array of images in numpy format
        :param dataset_ann: bool list of anomalies
        :param imgs_metadata: list of data points saved in the csv file (metadata of the dataset)
        :return:
        """
        with h5py.File(local_path_out + filename + '.hdf5', 'w') as f:
            # Dataset optimized for 2 bytes integer
            f.create_dataset('datacube_images', data=dataset, dtype='i2', compression="gzip",
                             compression_opts=9)
            # Anomaly in bool format
            f.create_dataset('datacube_anomaly', data=dataset_ann, dtype='i1', compression="gzip",
                             compression_opts=9)
            # Complete annotations of each image in the dataset
            # f.create_dataset('datacube_annotations', data=np.array(img_data), compression="gzip",
            #                  compression_opts=9)

        # with open(local_path_out + filename + '.metadata', 'w') as fd:
        #     index = 0
        #     for line in imgs_metadata:
        #         fd.write(str(index) + ', ' + ', '.join(line) + '\n')
        #         index += 1

        DatasetGenerator.write_dictionary_to_file(imgs_metadata, local_path_out + filename, use_index=True )

    @staticmethod
    def write_dictionary_to_file(dictionary, file_path, use_index=True):
        """

        :param dictionary: imgs_metadata
        :param file_path:
        :param use_index:
        :return:
        """
        with open(local_path_out + file_path + '.metadata', 'w') as fd:
            index = 0
            for line in dictionary:
                if use_index:
                    fd.write(str(index) + ', ' + ', '.join(line) + '\n')
                    index += 1
                else:
                    fd.write('.join(line)' + '\n')

    @staticmethod
    def test_write_metadata(self):
        imgs_metadata = FileHandler.listSelectedPoints()
        with open(local_path_out + "test" + '.metadata', 'w') as fd:
            index = 0
            for line in imgs_metadata:
                fd.write(str(index) + ', ' + ', '.join(line) + '\n')
                index += 1

    @staticmethod
    def print_custom_progressbar(index, mini_batch, mini_batch_index, n_batches):
        printProgressBar(index + 1, mini_batch,
                         prefix='Batch [' + str(mini_batch_index + 1) + '/' + str(n_batches) + '] - Progress:',
                         suffix='Complete', length=50)


    def generate_dataset_one_file(self, mini_batch, dimensions=None, output_file_name="dataset"):
        """
        Method to generate a dataset from the selected point list saved in the csv file. The current implementation
        works with complete images and not with specifics ROIs (regions of interest).
        Note: new method to support different size of images
        :param mini_batch: number of images per file
        :param dimensions: list of dimension of the images [width, height, n_channels]. If this parameter is omitted,
        te dimension of the first loaded image will be used.
        :param output_file_name: base name of the output files
        :return:  None
        """
        # --------------------------------------------------------------------------------------------------------------
        # UNCOMMENT THE FOLLOWING BLOCK WHEN WORKING WITH A NORMAL DATASET
        logger = open(local_path_dataset_log, "w")
        imgs_metadata = FileHandler.listSelectedPoints()

        np.random.shuffle(imgs_metadata)    # shuffling the dataset with normal distribution
        DatasetGenerator.write_dictionary_to_file(imgs_metadata, local_path_out + "sorted_dataset", use_index=True)
        # print(imgs_metadata)

        n_images = len(imgs_metadata)
        cnt_img = -1

        print("N: ", len(imgs_metadata), "-- just one file -- \n")

        # generating training dataset files ----------------------------------------------------------------------------

        # generating the first N-1 mini batches for TRAINING

        filename = output_file_name
        logger.write("*** File name: " + filename + '\n')

        cnt_img = self.process_multiple_datacubes(imgs_metadata, dimensions, filename,
                                                  1, 0, cnt_img, n_images, logger)



        print("Amount of processed images: ", cnt_img+1)
        print(" --- Process Complete! ----")


    def generate_dataset(self, mini_batch, dimensions=None, output_file_name="dataset"):
        """
        Method to generate a dataset from the selected point list saved in the csv file. The current implementation
        works with complete images and not with specifics ROIs (regions of interest).
        Note: new method to support different size of images
        :param mini_batch: number of images per file
        :param dimensions: list of dimension of the images [width, height, n_channels]. If this parameter is omitted,
        te dimension of the first loaded image will be used.
        :param output_file_name: base name of the output files
        :return:  None
        """
        # --------------------------------------------------------------------------------------------------------------
        # UNCOMMENT THE FOLLOWING BLOCK WHEN WORKING WITH A NORMAL DATASET
        # logger = open(local_path_dataset_log, "w")
        # imgs_metadata = FileHandler.listSelectedPoints()
        #
        # np.random.shuffle(imgs_metadata)    # shuffling the dataset with normal distribution
        # DatasetGenerator.write_dictionary_to_file(imgs_metadata, local_path_out + "sorted_dataset", use_index=True)
        # # print(imgs_metadata)
        #
        # n_images = len(imgs_metadata)
        # cnt_img = -1
        #
        # n_train = int(n_images * 0.7)  # total of images for the training set
        # n_test = n_images - n_train  # total of images for the test set


        # --------------------------------------------------------------------------------------------------------------
        # ADDED TO AUGMENT THE TRAINING DATASET
        logger = open(local_path_dataset_log, "w")
        cnt_img = -1
        training_set_augmented, test_set = get_expanded_training_set()
        n_train = len(training_set_augmented)
        n_test = len(test_set)
        # n_images = n_train + n_test
        imgs_metadata = training_set_augmented + test_set
        # --------------------------------------------------------------------------------------------------------------
        print("N: ", len(imgs_metadata), ", Train: ", n_train, "Test: ", n_test)

        # generating training dataset files ----------------------------------------------------------------------------
        n_batches = int(np.ceil(n_train / mini_batch))
        print("Total amount of batches for training: ", n_batches)

        # generating the first N-1 mini batches for TRAINING
        for mini_batch_index in range(int(np.floor(n_train / mini_batch))):
            filename = output_file_name + '_train_' + str(mini_batch_index)
            logger.write("*** train mini-batch: " + filename + '\n')

            # TODO solo pasar los metadatos necesarios y no el archivo completo
            cnt_img = self.process_multiple_datacubes(imgs_metadata, dimensions, filename,
                                                      n_batches, mini_batch_index, cnt_img, mini_batch, logger)

        # generating the last training mini-batch
        filename = output_file_name + '_train_' + str(int(np.floor(n_train / mini_batch)))
        n_last_mini_batch_imgs = n_train % mini_batch

        # TODO arreglar el orden de los parámetros y los n_batches
        cnt_img = self.process_multiple_datacubes(imgs_metadata, dimensions, filename,
                                                  n_batches, n_batches-1, cnt_img, n_last_mini_batch_imgs, logger)

        # generating testing dataset files ------------------------------------------------------------------------
        n_batches = int(np.ceil(n_test / mini_batch))
        print("Total amount of batches for testing: ", n_batches)

        # generating the first N-1 mini batches for TESTING
        for mini_batch_index in range(int(np.floor(n_test / mini_batch))):
            filename = output_file_name + '_test_' + str(mini_batch_index)
            logger.write("*** test mini-batch: " + filename + '\n')

            cnt_img = self.process_multiple_datacubes(imgs_metadata, dimensions, filename,
                                                      n_batches, mini_batch_index, cnt_img, mini_batch, logger)

        # generating the last testing mini-batch
        filename = output_file_name + '_test_' + str(int(np.floor(n_test / mini_batch)))
        n_last_mini_batch_imgs = n_test % mini_batch

        cnt_img = self.process_multiple_datacubes(imgs_metadata, dimensions, filename,
                                                  n_batches, n_batches-1, cnt_img, n_last_mini_batch_imgs, logger)

        print("Amount of processed images: ", cnt_img+1)
        print(" --- Process Complete! ----")


    def process_multiple_datacubes(self, imgs_metadata, dimensions, filename,
                                   n_batches, mini_batch_index, cnt_img, mini_batch_size, logger):
        """
        TODO english description
        función para no reescribir código. Lee múltiples imágenes de hipercubos basados en el ordenamiento aleatorio
        de los metadatos de hypercubos almacenado en el csv. Esta función solo alamacena un conjunto de imágenes
        determinado por el mini-batch size en un archivo con extensión hdf5.
        :param imgs_metadata: metadata list of all datacubes saved in the csv file
        :param dimensions: dimensions of the output datacubes
        :param filename: output name of the dataset mini-batch file
        :param mini_batch_size: amount of images in the current mini-batch
        :param mini_batch_index: current [training | test] mini-batch index
        :param n_batches: number of total batches   #TODO interchange position
        :param cnt_img: current count of processed images
        :param logger: file descriptor to log all the events
        :return: current count of processed images
        """

        dataset = self.allocate_mem_dataset(mini_batch_size, dimensions)
        dataset_ann = np.zeros([mini_batch_size, 1])

        batch_offset = cnt_img + 1

        for index in range(mini_batch_size):
            cnt_img = cnt_img + 1
            logger.write("  [" + str(index) + "]" + str(imgs_metadata[cnt_img]) + '\n')

            # reading all the wavelengths of one hyper-spectral image
            img_layers = DatasetGenerator.read_hypercube(imgs_metadata[cnt_img], dimensions)
            dataset[index] = img_layers

            dataset_ann[index] = 1 if imgs_metadata[cnt_img][3] == p_anomaly else 0

            self.print_custom_progressbar(index, mini_batch_size, mini_batch_index, n_batches)

        logger.write("--- Complete dataset annotations: " + '\n')
        logger.write(str(dataset_ann.T) + '\n')

        print(local_path_out + filename + '.hdf5' + '\n')

        self.write_dataset_to_file(filename, dataset, dataset_ann,
                                   imgs_metadata[batch_offset : batch_offset + mini_batch_size ])

        return cnt_img

    @staticmethod
    def read_hypercube(img_metadata, output_dimension):
        """
        reading all the wavelengths of one hyper-spectral image
        this is a customizable/hard-coded method to support multiple size images.

        TODO improve generalization parameters
        TODO manage exceptions when dimensions mismatch
        TODO add functionality to select wavelengths
        :param img_metadata: metadata of the current image to be read
        :param output_dimension: output dimension list [LINES, SAMPLES, BANDS]
        :return: [LINES, SAMPLES, BANDS] in floating point format
        """
        full_image = False

        if output_dimension is None:
            output_dimension = DEFAULT_DIMENSIONS
            full_image = True

        # special case where some images have a specific dimension that mismatch with current dimensions of the
        # acquisition.
        x_offset_start = 0
        if REGEX_IMG_201904 in img_metadata[5]:     # check if it's an old image
            x_offset_start = 40 # starting offset

        img_layers = np.zeros(output_dimension)

        # reading all the wavelengths
        for band in range(0, output_dimension[2]):
            # print(img_metadata[5] + " -- band: " + str(band + 1))
            data_read = ReadBilFile(img_metadata[5], band + 1)

            if full_image is True:
                img_layers[:, :, band] = data_read[x_offset_start:, :]
            else:
                # TODO improve how to get roi dimensions from the interface
                x1, y1, x2, y2 = DatasetGenerator.check_limits(int(img_metadata[1]), int(img_metadata[2]),
                                                               DEFAULT_ROI_DIMENSION_W, DEFAULT_ROI_DIMENSION_H,
                                                               SAMPLES, LINES)
                img_layers[:, :, band] = data_read[y1:y2, x1:x2]

        return img_layers


    @staticmethod
    def check_limits(x, y, w, h, x_limit, y_limit):
        """
        Crop an image using as reference the center point with dimensions w and h
        :param x: x coordinate of the center
        :param y: y coordinate of the center
        :param w: width of the cropped image
        :param h: height of the cropped image
        :param x_limit:
        :param y_limit:
        :return:
        """
        # x1 and y1 correspond to the left upper corner
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        # x1 and y1 correspond to the right lower corner
        x2 = x1 + w
        y2 = y1 + h
        # checking boundaries
        if x1 < 0:
            x1 = 0
            x2 = w
        if y1 < 0:
            y1 = 0
            y2 = h
        if x2 > x_limit:
            x1 = x_limit - w
        if y2 > y_limit:
            y1 = y_limit - h

        return x1, y1, x2, y2


def ReadBilFile(name, selected_band):
    """
    Funtion to read a datacube image

    Arguments:
    name            -- complete path with the name of the file
    selected_band   -- selected band to be read

    Returns:
    data            -- image in array format of the selected band
    """
    data = None     # default initialization for data read

    gdal.GetDriverByName('EHdr').Register()
    img = gdal.Open(name)
    # if img is None:
    #     print("Error opening the current file: " + name)
    # else:
    #     band = img.GetRasterBand(selected_band)
    #     data = band.ReadAsArray()
    band = img.GetRasterBand(selected_band)
    data = band.ReadAsArray()
    return data


def GetParameter(content, parameterName):
    """
    Get one parameter from the header file

    Arguments:
    content         -- content of the file
    parameterName   -- name of the parameter. Parameters available:
                    --  bands, lines, samples, framerate

    Returns:
    param           -- current value of the parameter
    """
    expression = parameterName + " ="
    index = content.find(parameterName) + len(expression) + 1
    param = content[index: content.find("\n", index)]
    param = int(param)

    return param


def get_parameters(img_path_complete):
    """
    Get the parameters of an hyper-spectral image from file.
    :param img_path_complete: complete path of the bil file to get the parameters. Ex. <path>/img.bil
    :return: array of totals of [lines, samples, bands]
    """
    file_info = img_path_complete + dcube_h  # header of the datacube image

    in_file = open(file_info, "rt")  # open the header file for reading text data
    info_file_content = in_file.read()  # read the entire file into a string variable
    in_file.close()  # close the file

    # regex for: lines, samples, bands
    regex = re.compile(r'.*lines\s*=\s*(\d+)\n.*'
                       r'.*samples\s*=\s*(\d+)\n.*'
                       r'.*bands\s*=\s*(\d+)\n.*', re.I)

    params = regex.findall(info_file_content)

    if params is not None:
        lines = int(params[0][0])
        samples = int(params[0][1])
        bands = int(params[0][2])

        return [lines, samples, bands]
    else:
        return None


def hyperspectralImageReadExample():
    # name of the file
    file_name = 'ACEITE 1.bil'
    # selected band to display in the first test
    sel_band = 160

    # relative or absolute path of the directory to read files
    file_path = '../img/'
    # default extension of the header
    data_ext = '.hdr'

    file_path_complete = file_path + file_name
    file_info = file_path_complete + data_ext

    # Read and display just one wavelength of the datacube image [1st test]
    img_read = ReadBilFile(file_path_complete, sel_band)

    in_file = open(file_info, "rt")  # open the header file for reading text data
    contents = in_file.read()  # read the entire file into a string variable
    in_file.close()  # close the file

    bands = GetParameter(contents, "bands")
    lines = GetParameter(contents, "lines")
    samples = GetParameter(contents, "samples")

    print("bands: ", bands)
    print("lines: ", lines)
    print("samples: ", samples)
    print("Each band consist of: (", lines, "x", samples, ")")

    img_layers = np.zeros((bands, lines, samples))

    # reading all the different wavelengths
    for j in range(0, bands):
        img_layers[:, :, j] = ReadBilFile(file_path_complete, j + 1)

    # some extra information about the data
    max_value = img_layers.max()
    print("max value present: ", max_value)
    bits_needed = np.log(max_value) // np.log(2) + 1
    print("bits needed to represent data: ", bits_needed)


def hyperspectral_image_read(file_path_complete, normalized=False):
    """

    :param file_path_complete:
    :param normalized:
    :return:
    """
    n_bits = 15

    file_info = file_path_complete + dcube_h

    in_file = open(file_info, "rt")  # open the header file for reading text data
    contents = in_file.read()  # read the entire file into a string variable
    in_file.close()  # close the file

    bands = GetParameter(contents, "bands")
    lines = GetParameter(contents, "lines")
    samples = GetParameter(contents, "samples")

    img_layers = np.zeros((lines, samples, bands))

    # reading all the different wavelengths
    for j in range(0, bands):
        img_layers[:, :, j] = ReadBilFile(file_path_complete, j + 1)

    if normalized is True:
        img_layers = img_layers / np.power(2, n_bits) - 1

    return img_layers



def use_case_generate_complete_imgs():
    """
    Use case to generate a dataset based on current ROI dimensions.
    """
    import time

    data = DatasetGenerator()
    t_start = time.time()

    mini_batch_size = 64

    data.generate_dataset(mini_batch_size, dimensions=None, output_file_name="dataset")

    t_end = time.time()

    print("Elapsed time: ", t_end - t_start, " [s]")
    print("Elapsed time: ", (t_end - t_start) / 60, " [min]")



def use_case_generate_rois():
    """
    Use case to generate a dataset based on current ROI dimensions.
    """
    import time

    data = DatasetGenerator()
    t_start = time.time()

    rois_per_file = 2048

    # data.generate_dataset(rois_per_file,
    #                       dimensions=[DEFAULT_ROI_DIMENSION_H, DEFAULT_ROI_DIMENSION_W, 168],
    #                       output_file_name="dataset_roi_40_40")

    data.generate_dataset_one_file(rois_per_file,
                                   dimensions=[DEFAULT_ROI_DIMENSION_H, DEFAULT_ROI_DIMENSION_W, 168],
                                   output_file_name="dataset_roi_4040")

    t_end = time.time()

    print("Elapsed time: ", t_end - t_start, " [s]")
    print("Elapsed time: ", (t_end - t_start) / 60, " [min]")


def __poc_clean_dataset():
    """
    Proof of concept to remove the duplicated information in the dataset due to double capture of some images rotated 90°
    :return:
    """

    points = FileHandler.listSelectedPoints()
    filtered_points = []

    regex = re.compile(r'\w*(\d+)')

    for point in points:
        var = point[0]
        result = regex.findall(var)
        if result is not None:
            index = int(result[0])
            if index % 2:
                print(index, " - impar")
            else:
                print(index, " - par")



if __name__ == '__main__':
    """

    """

    # use_case_generate_complete_imgs()

    # to generate all ROIs:
    use_case_generate_rois()

    # PoC
    # __poc_clean_dataset()
