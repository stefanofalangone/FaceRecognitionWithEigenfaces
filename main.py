from DatasetLoader import DatasetLoader
from FaceSpace import FaceSpace
from Utility import showImage
import numpy as np

dataset = DatasetLoader()
training_set = dataset.getTrainingSet()
test_set = dataset.getTestSet()
print("training set imgs ", training_set[0, :].size)
print("test set imgs ", test_set[0, :].size)
face_space = FaceSpace(training_set)

"""print("row 2", training_set[1, :])
print("sum is ", np.sum(training_set[1, :])/10.0 )
print("centroid element ", face_space.centroid[1])
print("centroid is\n")"""

"""showImage(face_space.centroid)

for i in range(training_set[: , 0].size - 1):
    showImage(training_set[: , i])"""