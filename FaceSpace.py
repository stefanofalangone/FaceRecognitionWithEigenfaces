import numpy as np

class FaceSpace:

    def __init__(self, training_set):
        self.training_set = training_set
        #print(training_set)
        self.computeCovarianceMatrix()

    def computeCovarianceMatrix(self):
        AT_A = np.dot(self.training_set.T, self.training_set)
        U, D, V_T = np.linalg.svd(AT_A)
        #print("eigenvalues ", D)
        eigenvectors = np.dot( self.training_set, V_T.T )
        print("rows are "+ str(eigenvectors[:,0].size) + " cols dim are " + str(eigenvectors[0, :].size))
