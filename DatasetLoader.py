import matplotlib.pyplot as plt
import numpy as np

class DatasetLoader:

    path = "/"
    n_images = 400
    width_images = 92
    height_images = 112
    TrainingSet = None
    TestSet = None

    def __init__(self):
        print("create the dataset Loader\n")
