from DatasetLoader import DatasetLoader
from FaceSpace import FaceSpace
from Utility import showImage
from Utility import createRandomImage
import numpy as np

"""a = np.arange(3)
print(np.linalg.norm(a)**2)"""

dataset = DatasetLoader()
training_set = dataset.getTrainingSet()
test_set = dataset.getTestSet()
training_set_labels = dataset.getTrainingSetLabels()
test_set_labels = dataset.getTestSetLabels()
print("training set imgs ", training_set[0, :].size)
print("test set imgs ", test_set[0, :].size)
face_space = FaceSpace(training_set, training_set_labels)

image_without_face_0 = np.zeros(10304)
image_without_face_255 = np.ones(10304)*255
random_image = createRandomImage(10304)
print(image_without_face_255)
print(random_image)
print(face_space.findMaximumProjectionError(training_set))
#face_space.testImageRecognition(test_set[:, 0])
result = face_space.testFaceDetection(test_set[:, 36])
#face_space.calculateTestsetAccuracy(test_set, test_set_labels)

print("Does image contain a face? ", result)
#for i in range(20):
    #showImage(face_space.eigenface_basis[:, i])
"""print("row 2", training_set[1, :])
print("sum is ", np.sum(training_set[1, :])/10.0 )
print("centroid element ", face_space.centroid[1])
print("centroid is\n")"""

"""showImage(face_space.centroid)

for i in range(training_set[: , 0].size - 1):
    showImage(training_set[: , i])"""