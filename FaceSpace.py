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
        while(D[i] > self.threshold and i<150):
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
        correct_predictions = 0
        total = 0
        correct_class = 0
        for i in range( test_set[0, :].size ):
            prediction = self.testImageRecognition(test_set[:, i])
            if i%1 == 0: correct_class = correct_class + 1
            print("prediction for image i of test = ", i, "is ", prediction, "correct class is ", correct_class)
            if correct_class == prediction: correct_predictions = correct_predictions + 1
            total = total + 1
        print("correct prediction / total ", correct_predictions/total)

    def testImageRecognition(self, input_image):
      image_chosen = 0
      #print("input image  shape ", np.shape(input_image) )
      #print("centroid  shape ", np.shape(self.centroid) )
      input_image = input_image.reshape(input_image.size , 1)
      input_image = ( input_image - self.centroid )
      #print("input image after centroid shape ", np.shape(input_image) )
      #print("input image dimension ", self.centroid , "input image ", input_image.size )
      cluster_similarity = np.zeros( len(self.centroid_per_classes) )
      image_0 = np.asarray( self.projectData(input_image) ).reshape(-1)

      #print("input image projected in face space shape ", np.shape(image_0))
      for i in range( 1, len(self.centroid_per_classes)+1 ):
          cluster_i = self.centroid_per_classes[i]
          #print("cluster_i shape in facespace ", np.shape(cluster_i))
          #diff = image_0 - cluster_i
          cosine = np.dot(image_0, cluster_i) / (np.linalg.norm(image_0) * np.linalg.norm(cluster_i))
          cluster_similarity [i-1] = cosine
          # print("distance ", image_chosen, " and i ", i,  np.format_float_scientific( np.dot(diff, diff)) )
          #print("cosine ", image_chosen, " and i ", i, cosine)
      n = 3
      indices = (-cluster_similarity).argsort()[:n] + 1
      print("most likely clusters: ", indices)
      # print(self.training_set_projection[:, 0].size)
      """for i in range(image_chosen, image_chosen+14):
          showImage( self.training_set[:, i] )"""
      return indices[0]

    def testFaceDetection(self, input_image):
        projection_error_square = self.computeProjectionErrorSquare(input_image)

        print("Projection Error Square: ", projection_error_square)
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
        for i in range(1,len(self.training_set_labels)+1):
            start = end
            end = start + len(self.training_set_labels[i])
            data_ith_class = self.training_set_projection[:, start:end]
            centroids_list[i] = np.concatenate(self.calculateCentroid(data_ith_class))
        self.centroid_per_classes = centroids_list