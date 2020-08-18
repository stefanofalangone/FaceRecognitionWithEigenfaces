import matplotlib.pyplot as plt
import numpy as np

class DatasetLoader:

    path = "/"
    n_directories = 0
    n_images_per_directory = 0
    width_images = 0
    height_images = 0
    training_set = None
    test_set = None

    def __init__(self):
        print("create the dataset Loader\n")
        self.setupImgFormat(self, 40, 10, 92, 112)

    def setupImgFormat(self, n_directories, n_images_per_directory, width_images, height_images):
        self.n_directories = n_directories
        self.n_images_per_directory = n_images_per_directory
        self.width_images = width_images
        self.height_images = height_images