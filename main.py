from DatasetLoader import DatasetLoader
from FaceSpace import FaceSpace
from Utility import showImage
from Utility import createRandomImage
import numpy as np

path_dataset_yale = "./datasetYale/Yale_Cropped_Dataset/"
dataset = DatasetLoader( path_dataset_yale )
training_set = dataset.getTrainingSet()
test_set = dataset.getTestSet()
training_set_labels = dataset.getTrainingSetLabels()
test_set_labels = dataset.getTestSetLabels()
print("training set imgs ", training_set[0, :].size)
print("test set imgs ", test_set[0, :].size)
face_space = FaceSpace(training_set, training_set_labels)
face_space.calculateTestsetAccuracy(test_set, test_set_labels)
showImage(training_set[:,0], 112, 92)
print("\nERROR thresold", np.format_float_scientific(face_space.findMaximumProjectionError(training_set)))
print("error in 36 image test set", np.format_float_scientific(face_space.computeProjectionErrorSquare(test_set[:, 36])))

#print("Does image contain a face? ", result)

image_without_face_0 = np.zeros(10304)
image_without_face_255 = np.ones(10304)*255
random_image = createRandomImage(10304)
"""showImage(image_without_face_0, 112, 92)
showImage(image_without_face_255, 112, 92)
showImage(random_image, 112, 92)"""

print("error committed on 0..0 image ", np.format_float_scientific(face_space.computeProjectionErrorSquare(random_image)))
result = face_space.testFaceDetection(random_image)
print("Does image contain a face? ", result)
#for i in range(20):
    #showImage(face_space.eigenface_basis[:, i], 112, 92)"""
"""print("row 2", training_set[1, :])
print("sum is ", np.sum(training_set[1, :])/10.0 )
print("centroid element ", face_space.centroid[1])
print("centroid is\n")"""

"""showImage(face_space.centroid, 112, 92)

for i in range(training_set[: , 0].size - 1):
    showImage(training_set[: , i], 112, 92)"""