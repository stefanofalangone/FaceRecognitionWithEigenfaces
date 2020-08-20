import numpy as np
from Utility import showImage

class FaceSpace:

    threshold = 10**-3
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

        print(self.centroid_per_classes[5])

    def computeEigenfaceBasis(self):
        AT_A = np.dot(self.training_set.T, self.training_set)
        U, D, V_T = np.linalg.svd(AT_A)
        #print("eigenvalues ", D)
        #eigenvectors = np.dot( self.training_set, V_T.T )
        eigenvectors = V_T.T
        eigenvectors = self.calculateEigenvectors(eigenvectors)
        i = 0
        current_vector = []
        while(D[i] > self.threshold and i<279):
                current_vector.append( eigenvectors[: , i] )
                #print("division is ", D[i] / D[ D.size - 1 ])
                #print("D[i] and last are ", D[i], D[ D.size - 1 ])
                i = i + 1
        #print("i is ", i)
        self.eigenface_basis = np.stack(current_vector, axis = -1)

        print("rows are "+ str(self.eigenface_basis[:,0].size) + " cols dim are " + str(self.eigenface_basis[0, :].size))

    def calculateEigenvectors(self, eigenvectors):
        ret = []
        for i in range( eigenvectors[0].size ):
            v = np.array(eigenvectors[:, i])
            sum = self.training_set * v
            sum = np.sum(sum, axis = -1)
            ret.append(sum)
        eigen = np.stack( ret, axis = -1 )
        print("eigen dimensions: rows ", eigen[:, 0].size, "cols:", eigen[0].size )
        return eigen

    def projectTrainingSet(self):
        result = []
        for i in range(self.training_set[0, :].size):
            result.append(self.projectData(self.training_set[:, i]))
        self.training_set_projection = np.stack(result, axis = -1)

        for i in range( self.training_set[0, :].size ):
            image_chosen = 14
            diff = self.training_set_projection[ : , image_chosen ] - self.training_set_projection[: , i]
            #print("distance ", image_chosen, " and i ", i,  np.format_float_scientific( np.dot(diff, diff)) )
        #print(self.training_set_projection[:, 0].size)
        #showImage( self.training_set[:, image_chosen] )

    def projectData(self, image):
        return np.dot(self.eigenface_basis.T, image)

    def centerData(self, data):
        if (self.centroid == None):
            centroid = self.calculateCentroid(data)
        else:
            centroid = self.centroid
        return data - centroid

    def calculateCentroid(self, data):
        number_of_data = data[0, :].size
        centroid = data.sum(axis=1) / number_of_data
        self.centroid = centroid.reshape(centroid.size, 1)
        return self.centroid

    def calculateCentroidForEachClass(self):
        centroids_list = []
        end = 0
        for i in range(1,len(self.training_set_labels)+1):
            start = end
            end = start + len(self.training_set_labels[i])
            data_ith_class = self.training_set_projection[:, start:end]
            centroids_list.append(np.concatenate(self.calculateCentroid(data_ith_class)))
        self.centroid_per_classes = centroids_list