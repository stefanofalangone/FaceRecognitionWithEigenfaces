import matplotlib.pyplot as plt
import numpy as np

def showImage( vectorialized_image, height, width ):
    plt.imshow(vectorialized_image.reshape((height, width)), cmap='gray')
    plt.show()

def createRandomImage(size):
    return np.random.randint(0, 256, (size))
