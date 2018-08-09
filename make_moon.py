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
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Spectral)
        plt.title("Data distribution of 2 classes [0, 1]")
        plt.show()

    def _make_batch(self):
        batch = np.array(np.zeros(shape=self.batch_size))
        labels = np.array(np.zeros(shape=self.batch_size))

        def make_one_hot(label):
            return (np.arange(self.num_class) == labels[:, None]).astype(np.integer)

        if self.pos + self.batch_size <= self.data_length:
            batch = self.X[self.pos: self.pos + self.batch_size]
            labels = self.y[self.pos: self.pos + self.batch_size]
            self.pos = self.pos + self.batch_size
        else:
            batch = self.X[self.pos: self.data_length]
            labels = self.y[self.pos: self.data_length]

            batch = np.vstack((batch, self.X[0: self.batch_size - (self.data_length - self.pos)]))
            labels = np.append(labels, self.y[0: self.batch_size - (self.data_length - self.pos)])
            self.pos = self.batch_size - (self.data_length - self.pos)
        assert len(batch) == self.batch_size
        assert len(labels) == self.batch_size
        return batch, labels, make_one_hot(labels)

    def _softmax(self, input_array):
        if len(input_array.shape) > 1:
            c = np.max(input_array, axis=1)
            temp_value = list(map(lambda x_val, c_val: x_val - c_val, input_array, c))
            numerator = np.exp(temp_value)
            denomater = np.sum(numerator, axis=1).reshape(-1, 1)
            result = numerator / denomater
        else:
            c = np.max(input_array, axis=0)
            numerator = np.exp(input_array - c)
            denomater = np.sum(numerator)
            result = numerator / denomater
        assert result.shape == input_array.shape
        return result

    def _forward_pass(self):
        self.batch, self.labels, self.one_hot_label = self._make_batch()
        self.h1 = np.dot(self.batch, self.weight_1) + self.b1
        # activation function
        self.z1 = np.tanh(self.h1)
        self.logits = np.dot(self.z1, self.weight_2) + self.b2
        self.probs = self._softmax(self.logits)
        pred_label = np.argmax(self.probs, axis=1)
        accuracy = np.mean(np.equal(pred_label, self.labels))

        # calculate loss of model
        self.loss = - np.sum(list(map(lambda x, y: x[y], np.log(self.probs), self.labels)))
        self.loss += 0.5 * self.regulation * (np.sum(np.square(self.weight_2)) + np.sum(np.square(self.weight_1)))
        self.loss /= self.num_class
        return self.logits, self.loss, accuracy

    def _back_progation(self):
        delta1 = self.probs
        for i in range(0, len(self.labels)):
            delta1[i, self.labels[i]] -= 1
        delta2 = (1 - np.power(self.z1, 2)) * np.dot(delta1, self.weight_2.T)

        # solve gradient for weights and bias
        delta_b2 = np.sum(delta1, axis=0, keepdims=True)
        delta_b1 = np.sum(delta2, axis=0)

        delta_weight1 = np.dot(self.batch.T, delta2)
        delta_weight2 = np.dot(self.z1.T, delta1)


        # add regulation term
        delta_weight2 += self.regulation * self.weight_2
        delta_weight1 += self.regulation * self.weight_1

        # gradient descent and update parameters
        self.weight_1 += -self.lr * delta_weight1
        self.b1 += -self.lr * delta_b1
        self.weight_2 += -self.lr * delta_weight2
        self.b2 += -self.lr * delta_b2

    def _save_model(self, save_path):
        model = {"w1": self.weight_1,
                 "w2": self.weight_2,
                 "b1": self.b1,
                 "b2": self.b2}

        with open(save_path, "wb") as model_file:
            pickle.dump(model, model_file)

    def train(self, save_model_path):
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        for num_step in range(0, self.config.num_steps):
            # forwardpropagation
            _, loss, accuracy = self._forward_pass()
            if num_step % 1000 == 0:
                print("Current loss is %.6f, accuracy is %.6f at step %d: " %(loss, accuracy, num_step))
                modle_path = os.path.join(save_model_path, "nn_step_" + str(num_step) + "_.pkl")

                self._save_model(modle_path)
            # backproagation, update loss value
            self._back_progation()

    def test(self, model_name):
        with open(os.path.join("./check_point", model_name), "rb") as f:
            model = pickle.load(f)

        print(model["w1"].shape, model["b1"].shape, model["w2"].shape, model["b2"].shape)
        x_coo = []
        y_coo = []
        label = []
        for x in np.linspace(-2.0, 2.5, 200):
            for y in np.linspace(-2.0, 2.5, 200):
                input = np.array([x, y])

                h1 = np.dot(input, model["w1"]) + model["b1"]
                sigma1 = np.tanh(h1)
                h2 = np.dot(sigma1, model["w2"]) + model["b2"]
                test_prob = self._softmax(h2)
                pred_label = np.argmax(test_prob)
                x_coo.append(x)
                y_coo.append(y)
                label.append(pred_label)

        plt.scatter(x_coo, y_coo, c=label, cmap=plt.cm.Paired)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Spectral)
        plt.title("Decision boundary of model %s" %model_name)
        plt.show()




if __name__ == "__main__":
    data = DataClassfication(0, batch_size=100, n_feature=10, regulation_coef=0.001, learning_rate=0.01)
    #data.train("./check_point")
    data.test("nn_step_0_.pkl")