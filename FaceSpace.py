import numpy as np
from Utility import showImage

class FaceSpace:

    threshold = 10**-3
    eigenface_basis = None
    centroid = None
    training_set_projection = None

    def __init__(self, training_set):
        self.training_set = self.centerData( training_set )
        self.computeEigenfaceBasis()
        self.projectTrainingSet()

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
        self.eigenface_basis = np.stack(current_vector, axis = -1)

        print("rows are "+ str(self.eigenface_basis[:,0].size) + " cols dim are " + str(self.eigenface_basis[0, :].size))
        #for i in range( self.eigenface_basis[0 , :].size ):
        #   showImage( self.eigenface_basis[:, i] )

    def calculateEigenvectors(self, eigenvectors):
        ret = []
        for i in range( eigenvectors[0].size ):
            v = np.array(eigenvectors[:, i])
            sum = self.training_set * v
            sum = np.sum(sum, axis = -1)
            ret.append(sum)
        eigen = np.stack( ret, axis = -1 )
        print("eigen dimensions: rows ", eigen[:, 0].size, "cols:", eigen[0].size )
        print("eigen ", eigen)
        return eigen

    def projectTrainingSet(self):
        result = []
        for i in range(self.training_set[0, :].size):
            result.append(self.projectData(self.training_set[:, i]))
        self.training_set_projection = np.stack(result, axis = -1)
        print("training set projection ", self.training_set_projection)
        print("training set projection first image ", self.training_set_projection[: , 0])
        print("training set projection 4 image ", self.training_set_projection[: , 3])

        for i in range( self.training_set[0, :].size ):
            diff = self.training_set_projection[: , 10] - self.training_set_projection[: , i]
            print("distance 10 and i ", i,  np.square( np.dot(diff, diff)) )

        for i in range(self.training_set[0, :].size):

            #dot = np.dot(self.training_set_projection[: , i]  , self.eigenface_basis.T  )
            dot = np.dot(self.training_set_projection.T  , self.eigenface_basis.T  )
            dot = dot / np.linalg.norm(dot)
            distance = np.linalg.norm(self.training_set[:, 0] - dot)
            #print("image of training ", i, "distance is ", distance)
        #showImage(self.training_set[:, 0])
        #showImage(dot)

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

