from __future__ import (absolute_import, division, print_function, unicode_literals)

import subprocess as sub
from .supervised_learner import SupervisedLearner
from .matrix import Matrix
import numpy as np


class PerceptronLearner(SupervisedLearner):

    best_weights = []
    weights = []
    labels = [] #0,1,2 for iris
    error_rate = []

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.best_weights = np.zeros((labels.value_count(0), features.cols + 1))
        self.weights = np.zeros((labels.value_count(0), features.cols + 1))
        empty_array = []
        #for i in range(labels.value_count(0)):
            #self.error_rate.append(empty_array)
        #f = open('output.txt', 'a')
        #print('-'*100)
        #print("Pattern        Bias   Target    Weight Vector       Net   Output   Change in Weight")
        #print('-'*100)
        for i in range(labels.value_count(0)):
            for j in range(features.cols):
                self.weights[i][j] = np.random.rand()*.1-.05
        learning_rate = .1
        num_correct = np.zeros(len(self.weights))
        times_unchanged = 0
        times_through = 0
        while times_unchanged < 5:
            changed = 0
            features.shuffle(labels)
            new_correct = np.zeros(len(self.weights))
            for i in range(features.rows):
                for x in range(len(self.weights)):#runs method for each weight
                    prediction_num = 0
                    label = 0 #makes all labels either one or zero for the sake of weight adjustment
                    result = 0
                    for j in range(features.cols):
                        #make prediction with weights
                        prediction_num += self.weights[x][j]*features.get(i,j)
                    prediction_num += self.weights[x][-1]
                    #compare prediction to reality
                    if prediction_num > 0:
                        result = 1 #result will be equal to the category number

                    #change if wrong
                    if x == labels.get(i,0):
                        label = 1
                    if result == label:
                        new_correct[x] += 1
                    else:
                        for j in range(features.cols):
                            delta = (label - result) * learning_rate * features.get(i,j)
                            self.weights[x][j] += delta
                        delta = (label - result) * learning_rate
                        self.weights[x][-1] += delta
                    #mark improvement for each Perceptron
                    self.error_rate.append(1 - new_correct[x]/labels.rows)
                    if new_correct[x] > num_correct[x]:
                        changed = 1
                        self.best_weights[x] = self.weights[x]
                        num_correct[x] = new_correct[x]
                        times_unchanged = 0
            if changed == 0:
                times_unchanged += 1
            #if not improving, change to end soon. Else, mark improvement
            times_through += 1
        self.weights = self.best_weights
        print("Weights are: ")
        print(np.around(self.weights,3))
        print("Error rate over time: ")
        print(np.around(self.error_rate,3))
        print("\nTimes through: ")
        print(times_through)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        self.labels = [0]
        del labels[:]
        predictions = []
        for x in range(len(self.weights)):
            prediction_num = 0
            for j in range(len(features)):
                #make prediction with weights
                prediction_num += self.weights[x][j]*features[j]
            prediction_num += self.weights[x][-1]
            predictions.append(prediction_num)
        #choose prediction with highest net value
        highest_net = predictions[0]
        prediction = 0 #our first prediction is the first value
        for x in range(len(predictions)):
            if predictions[x] > highest_net:
                highest_net = predictions[x]
                prediction = x
        self.labels = [prediction]
        labels += self.labels
