import matplotlib.pyplot as plt
import numpy as np

def showImage( vectorialized_image ):
    plt.imshow(vectorialized_image.reshape((112, 92)), cmap='gray')
    plt.show()

def createRandomImage(size):
    return np.random.randint(0, 256, (size))
