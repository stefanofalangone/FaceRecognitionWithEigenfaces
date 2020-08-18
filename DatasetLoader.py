import matplotlib.pyplot as plt
import numpy as np

"""This class shall be responsible of the correct reading of the 'dataset' directory.
    In our specific case, there are 40 directories (one for each subject) of 10 images each.
    Every image is 92x112 pixels of 8-bit grey scale. (Each img say this anyway)
"""
class DatasetLoader:

    path = "dataset/"
    n_directories = 0
    n_images_per_directory = 0
    width_images = 0
    height_images = 0
    should_check_pixel_format = True
    training_set = None
    test_set = None

    def __init__(self, path="dataset/"):
        print("create the dataset Loader\n")
        self.path = path
        self.setupDirectoryFormat(40, 10)
        self.setupImgFormat(92, 112)

    def setupDirectoryFormat(self, n_directories=40, n_images_per_directory=10):
        self.n_directories = n_directories
        self.n_images_per_directory = n_images_per_directory

    def setupImgFormat(self, width_img=92, height_img=112):
        self.width_images = width_img
        self.height_images = height_img
        should_check_pixel_format = False

    def load(self):
        for subj in range(1, 41, 1):
            for pic in range (1, 11, 1):
                print("subj ", subj, ". Pic ", pic)