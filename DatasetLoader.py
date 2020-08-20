
import numpy as np
import random

class DatasetLoader:

    path = "./dataset/"
    n_directories = 0
    n_images_per_directory = 0
    width_images = 0
    height_images = 0
    training_set = None
    test_set = None
    training_set_labels = None
    test_set_labels = None

    def __init__(self, path="./dataset/"):
        print("create the dataset Loader\n")
        self.path = path
        self.setupDirectoryFormat(40, 10)
        self.setupImgFormat(92, 112)
        self.training_set, self.test_set, self.training_set_labels, self.test_set_labels = self.extractTrainingsetTestset(70)

        #self.showImage(self.training_set[:,0])

    def setupImgFormat(self, width_img=92, height_img=112):
        self.width_images = width_img
        self.height_images = height_img

    def setupDirectoryFormat(self, n_directories=40, n_images_per_directory=10):
        self.n_directories = n_directories
        self.n_images_per_directory = n_images_per_directory


    def extractTrainingsetTestset(self, trainingPercentage):
        training_set = []
        test_set = []
        training_set_labels = {}
        test_set_labels = {}

        data_list = [i for i in range(1, self.n_images_per_directory+1)]
        number_of_training_images_per_directories = int((self.n_images_per_directory * trainingPercentage)/100)
        number_of_testing_images_per_directories = self.n_images_per_directory - number_of_training_images_per_directories

        for i in range(1, self.n_directories+1):
            test_list = random.sample(data_list, number_of_testing_images_per_directories)
            train_list = list(set(data_list) - set(test_list))
            training_set_labels[i] = train_list
            test_set_labels[i] = test_list
            path = self.path + 's'+str(i)+'/'

            for j in train_list:
                training_set.append(self.readPgm(path+str(j)+'.pgm'))

            for k in test_list:
                test_set.append(self.readPgm(path+str(k)+'.pgm'))

        return np.stack(training_set, axis=-1), np.stack(test_set, axis=-1), training_set_labels, test_set_labels


        vectorialized_image = []
        for k in range(height):
            for j in range(width):
                vectorialized_image.append(ord(pgmf.read(1)))

        return np.array(vectorialized_image, dtype='float')


    def readPgm(self, path):
        pgmf = open(path, 'rb')
        pgmf.readline()
        width, height = [int(i) for i in pgmf.readline().split()]
        max_val = int(pgmf.readline())
        assert max_val <= 255

        vectorialized_image = []
        for k in range(height):
            for j in range(width):
                vectorialized_image.append(ord(pgmf.read(1)))
        pgmf.close()

        return np.array(vectorialized_image, dtype='float')/255.0

    def getTrainingSet(self):
        return self.training_set

    def getTestSet(self):
        return self.test_set

    def getTrainingSetLabels(self):
        return  self.training_set_labels

    def getTestSetLabels(self):
        return  self.test_set_labels
