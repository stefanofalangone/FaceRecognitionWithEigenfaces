import numpy as np
from Utility import showImage

class FaceSpace:

    threshold = 10**-3
    eigenface_basis = None
    centroid = None

    def __init__(self, training_set):
        self.training_set = self.centerData( training_set )
        print( self.training_set )
        #showImage(self.training_set[:, 20])
        #showImage(training_set[:, 20])
        self.computeEigenfaceBasis()

    def computeEigenfaceBasis(self):
        AT_A = np.dot(self.training_set.T, self.training_set)
        U, D, V_T = np.linalg.svd(AT_A)
        print("eigenvalues ", D)
        eigenvectors = np.dot( self.training_set, V_T.T )

        i = 0
        current_vector = []
        while(D[i] > self.threshold):
                current_vector.append( eigenvectors[: , i] )
                #print("division is ", D[i] / D[ D.size - 1 ])
                #print("D[i] and last are ", D[i], D[ D.size - 1 ])
                i = i + 1
        self.eigenface_basis = np.stack(current_vector, axis = -1)

        print("rows are "+ str(self.eigenface_basis[:,0].size) + " cols dim are " + str(self.eigenface_basis[0, :].size))
        #for i in range( self.eigenface_basis[0 , :].size ):
        #    showImage( self.eigenface_basis[:, i] )

    def centerData(self, images):
        if (self.centroid == None):
            centroid = self.calculateCentroid(images)
        else:
            centroid = self.centroid
        return images - centroid

    def calculateCentroid(self, images):
        number_of_images = images[0, :].size
        centroid = images.sum(axis=1) / number_of_images
        self.centroid = centroid.reshape(centroid.size, 1)
        return self.centroid

