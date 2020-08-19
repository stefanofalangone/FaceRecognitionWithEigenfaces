import matplotlib.pyplot as plt
import numpy as np
import random

class DatasetLoader:

    path = "./dataset/"
    n_directories = 0
    n_images_per_directory = 0
    width_images = 0
    height_images = 0
    should_check_pixel_format = True
    training_set = None
    test_set = None

    def __init__(self, path="./dataset/"):
        print("create the dataset Loader\n")
        self.path = path
        self.setupDirectoryFormat(40, 10)
        self.setupImgFormat(92, 112)
        self.training_set, self.test_set = self.extractTrainingsetTestset(70)

        self.showImage(self.training_set[:,0])

    def setupImgFormat(self, width_img=92, height_img=112):
        self.width_images = width_img
        self.height_images = height_img
        should_check_pixel_format = False

    def setupDirectoryFormat(self, n_directories=40, n_images_per_directory=10):
        self.n_directories = n_directories
        self.n_images_per_directory = n_images_per_directory


    def extractTrainingsetTestset(self, trainingPercentage):
        training_set = []
        test_set = []

        data_list = [i for i in range(1, self.n_images_per_directory+1)]
        number_of_training_images_per_directories = int((self.n_images_per_directory * trainingPercentage)/100)
        number_of_testing_images_per_directories = self.n_images_per_directory - number_of_training_images_per_directories

        for i in range(1, self.n_directories+1):
            test_list = random.sample(data_list, number_of_testing_images_per_directories)
            train_list = list(set(data_list) - set(test_list))
            path = self.path + 's'+str(i)+'/'

            for j in train_list:
                training_set.append(self.readPgm(path+str(j)+'.pgm'))

            for k in test_list:
                test_set.append(self.readPgm(path+str(k)+'.pgm'))

        return np.stack(training_set, axis=-1), np.stack(test_set, axis=-1)

    def readPgm(self, path):
        pgmf = open(path, 'rb')
        assert pgmf.readline() == b'P5\n'
        width, height = [int(i) for i in pgmf.readline().split()]
        max_val = int(pgmf.readline())
        assert max_val <= 255

        vectorialized_image = []
        for k in range(height):
            for j in range(width):
                vectorialized_image.append(ord(pgmf.read(1)))

        return np.array(vectorialized_image, dtype='float')


    def getTrainingSet(self):
        return self.training_set

    def getTestSet(self):
        return self.test_set

    def showImage(self, vectorialized_image):
        plt.imshow(vectorialized_image.reshape((112, 92)), cmap='gray')
        plt.show()

    def load(self):
        for subj in range(1, 41, 1):
            for pic in range(1, 11, 1):
                print("subj ", subj, ". Pic ", pic)