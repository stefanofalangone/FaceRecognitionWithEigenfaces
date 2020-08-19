from DatasetLoader import DatasetLoader
from FaceSpace import FaceSpace

dataset = DatasetLoader()
training_set = dataset.getTrainingSet()
test_set = dataset.getTestSet()

face_space = FaceSpace(training_set)