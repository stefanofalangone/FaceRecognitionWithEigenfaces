import matplotlib.pyplot as plt

def showImage(vectorialized_image):
    plt.imshow(vectorialized_image.reshape((112, 92)), cmap='gray')
    plt.show()
