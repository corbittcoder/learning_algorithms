from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import math


class NeuralNetLearner(SupervisedLearner):
    """
    Nested Perceptrons with Logit functions
    """
    best_weights = []
    weights = [] #three dimensional array: number of perceptrons, number of rows, number of features
    labels = []
    error_rate = []
    NUM_LAYERS = 1
    NUM_NODES = 1 #nodes per layer

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        Use validation test set
        """
        NUM_INPUTS = len(features.cols + 1)
        self.NUM_NODES = 2 * math.log(NUM_INPUTS) #nodes = 2log(n)
        self.NUM_LAYERS = 1
        count = NUM_INPUTS
        while count >= 1:
            count = count / 10
            NUM_LAYERS += 1 #adds a layer for each order of magnitude, at least one hidden layer
        self.best_weights = np.zeros((NUM_NODES, NUM_LAYERS, NUM_INPUTS))
        self.weights = np.zeros((NUM_NODES, NUM_LAYERS, NUM_INPUTS))

        for i in range(NUM_NODES):
            for j in range(NUM_LAYERS):
                for k in range(NUM_INPUTS):
                    self.weights[i][j][k] = np.random.rand() * .1 - .05 #initializes array to random weights between +-.05

        features.shuffle(labels)
        validation_set_data = features[0:len(features.rows)*.18]
        features = features[len(features.rows)*.18:]
        validation_set_labels = labels[0:len(features.rows)*.18]
        labels = labels[len(features.rows)*.18:]
        LEARNING_RATE = .1
        times_unchanged = 0
        times_through = 0
        while times_unchanged < 5:
            changed = 0
            features.shuffle(labels)
            for i in range(features.rows):

    def check_prediction(self, net, label):
        """
        :type net: float(net value of weights times inputs)
        :type labels: float
        """
        return error_rate

    def weight_change(self, begin, end, error):
        """
        :type begin: int
        :type error: float
        error is the amount of error specific to that layer
        """

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels
