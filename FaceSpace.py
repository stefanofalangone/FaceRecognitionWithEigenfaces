import numpy as np
from Utility import showImage

class FaceSpace:

    threshold = 10**-3
    error_projection_threshold = 1.56*(10**25)
    eigenface_basis = None
    centroid = None
    centroid_per_classes = None
    training_set_projection = None

    def __init__(self, training_set, training_set_labels):
        self.training_set = self.centerData( training_set )
        self.training_set_labels = training_set_labels
        self.computeEigenfaceBasis()
        self.projectTrainingSet()
        self.calculateCentroidForEachClass()

    def computeEigenfaceBasis(self):
        AT_A = np.dot(self.training_set.T, self.training_set)
        U, D, V_T = np.linalg.svd(AT_A)
        #print("eigenvalues ", D)
        #eigenvectors = np.dot( self.training_set, V_T.T )
        eigenvectors = V_T.T
        eigenvectors = self.calculateEigenvectors(eigenvectors)
        i = 0
        current_vector = []
        while(D[i] > self.threshold and i< self.training_set[0 , :].size ):
                current_vector.append( eigenvectors[: , i] )
                #print("division is ", D[i] / D[ D.size - 1 ])
                #print("D[i] and last are ", D[i], D[ D.size - 1 ])
                i = i + 1
        #print("i is ", i)
        self.eigenface_basis = np.stack(current_vector, axis = -1)

        print("rows are "+ str(self.eigenface_basis[:,0].size) + " cols dim are " + str(self.eigenface_basis[0, :].size))

    """
        Determine linear combination of the M training set face images to form the eigenfaces
    """
    def calculateEigenvectors(self, eigenvectors):
        """ret = []
        for i in range( eigenvectors[0].size ):
            v = np.array(eigenvectors[:, i])
            sum = self.training_set * v
            sum = np.sum(sum, axis = -1)
            ret.append(sum)
        eigen = np.stack( ret, axis = -1 )"""
        eigen2 = np.dot(self.training_set, eigenvectors)
        """print("are the two eigens equal? ", np.array_equiv(eigen, eigen2))
        print("are the two eigens close? ", np.allclose(eigen, eigen2))
        print("FIRST EIGEN:")
        print(eigen[:, :])
        print("SECOND EIGEN")
        print(eigen2[:, :])
        print("eigen dimensions: rows ", eigen[:, 0].size, "cols:", eigen[0].size )"""
        return eigen2

    def projectTrainingSet(self):
        result = []
        for i in range(self.training_set[0, :].size):
            result.append(self.projectData(self.training_set[:, i]))
        self.training_set_projection = np.stack(result, axis = -1)

    def calculateTestsetAccuracy(self, test_set, test_set_labels):
        correct_predictions_cosine = 0
        correct_predictions_euclidean = 0
        total = 0
        for i in range( test_set[0, :].size ):
            prediction_cosine = self.testImageRecognitionWithCosine(test_set[:, i])
            prediction_euclidean = self.testImageRecognitionWithEuclideanDistance(test_set[:, i])
            correct_class = test_set_labels[i]
            print("prediction for image i of test = ", i, "is ", prediction_cosine, "correct class is ", correct_class)
            if correct_class == prediction_cosine: correct_predictions_cosine = correct_predictions_cosine + 1
            if correct_class == prediction_euclidean: correct_predictions_euclidean = correct_predictions_euclidean + 1
            total = total + 1
        print("COSINE correct prediction / total ", correct_predictions_cosine/total)
        print("EUCLIDEAN correct prediction / total ", correct_predictions_cosine/total)

    def testImageRecognitionWithCosine(self, input_image):
        cluster_similarity = self.computeCosineSimilarityForEachClass(input_image)
        n = 3
        indices = (-cluster_similarity).argsort()[:n] + 1
        print("[COSINE] most likely clusters: ", indices)
        return indices[0]

    def testImageRecognitionWithEuclideanDistance(self, input_image):
        cluster_similarity = self.computeEuclideanDistanceForEachClass(input_image)
        n = 3
        indices = (cluster_similarity).argsort()[:n] + 1
        print("[EUCLIDEAN] most likely clusters: ", indices)
        return indices[0]

    def computeCosineSimilarityForEachClass(self, input_image):
        input_image = input_image.reshape(input_image.size, 1)
        input_image = (input_image - self.centroid)
        cluster_similarity = np.zeros(len(self.centroid_per_classes))
        image_0 = np.asarray(self.projectData(input_image)).reshape(-1)
        for i in range(1, len(self.centroid_per_classes) + 1):
            cluster_i = self.centroid_per_classes[i]
            cosine = np.dot(image_0, cluster_i) / (np.linalg.norm(image_0) * np.linalg.norm(cluster_i))
            cluster_similarity[i - 1] = cosine
        return cluster_similarity

    def computeEuclideanDistanceForEachClass(self, input_image):
        input_image = input_image.reshape(input_image.size, 1)
        input_image = (input_image - self.centroid)
        cluster_distance = np.zeros(len(self.centroid_per_classes))
        image_0 = np.asarray(self.projectData(input_image)).reshape(-1)
        for i in range(1, len(self.centroid_per_classes) + 1):
            cluster_i = self.centroid_per_classes[i]
            diff = image_0 - cluster_i
            distance = np.linalg.norm(diff)**2
            cluster_distance[i - 1] = distance
        return cluster_distance

    def testFaceDetection(self, input_image):
        projection_error_square = self.computeProjectionErrorSquare(input_image)

        print("Projection Error Square: ", np.format_float_scientific(projection_error_square))
        return projection_error_square < self.error_projection_threshold

    def computeProjectionErrorSquare(self, input_image):
        input_image = input_image.reshape(input_image.size, 1)
        input_image = input_image - self.centroid
        eigenface_pattern_vectors = np.asarray(self.projectData(input_image)).reshape(-1)
        image_projected = np.dot(self.eigenface_basis, eigenface_pattern_vectors)
        image_projected = image_projected.reshape((image_projected.size, 1))
        difference = input_image - image_projected
        return np.linalg.norm(difference) ** 2

    def findMaximumProjectionError(self, set_of_images):
        errors_projection_list = []
        for i in range(set_of_images[0, :].size):
            image = set_of_images[:, i]
            projection_error_square = self.computeProjectionErrorSquare(image)
            errors_projection_list.append(projection_error_square)
        self.error_projection_threshold = 1.3 * max(errors_projection_list)
        return max(errors_projection_list)

    def projectData(self, image):
        return np.dot(self.eigenface_basis.T, image)

    def centerData(self, data):
        if (self.centroid == None):
            self.centroid = self.calculateCentroid(data)
        return data - self.centroid

    def calculateCentroid(self, data):
        number_of_data = data[0, :].size
        centroid = data.sum(axis=1) / number_of_data
        return centroid.reshape(centroid.size, 1)

    def calculateCentroidForEachClass(self):
        centroids_list = {}
        end = 0
        for i in range(1, max(self.training_set_labels)+1):
            start = end
            end = start + self.training_set_labels.count(i)
            data_ith_class = self.training_set_projection[:, start:end]
            centroids_list[i] = np.concatenate(self.calculateCentroid(data_ith_class))
        self.centroid_per_classes = centroids_list