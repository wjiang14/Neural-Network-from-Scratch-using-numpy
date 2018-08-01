import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pickle, os

class Configuration():
    def __init__(self):
        self.regulation = 0.4
        self.num_steps = 10000

class DataClassfication(object):
    def __init__(self, random_seed, batch_size, n_feature, regulation_coef, learning_rate):
        self.config = Configuration()
        np.random.seed(random_seed)
        self.X, self.y = make_moons(500, noise=0.20)
        self.batch_size = batch_size
        self.regulation = regulation_coef
        self.lr = learning_rate
        self.data_length = self.X.shape[0]
        self.num_class = self.X.shape[1]
        self.pos = 0
        if self.batch_size > self.data_length:
            raise Exception("Batch size is setted too large, batch: ", batch_size, "data lenght: ", self.data_length)

        # weight params in neural network
        self.weight_1 = np.random.uniform(size=[self.num_class, n_feature])
        self.b1 = np.zeros(shape=[batch_size, n_feature])
        self.weight_2 = np.random.uniform(size=[n_feature, self.num_class])
        self.b2 = np.zeros(shape=[self.batch_size, self.num_class])

    def plot_data(self):
        plt.scatter(data.X[:, 0], data.X[:, 1], c=data.y, cmap=plt.cm.Spectral)
        plt.title("Data distribution of 2 classes [0, 1]")
        plt.show()


if __name__ == "__main__":
    data = DataClassfication(0)

