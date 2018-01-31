from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import csv
import matplotlib.pyplot as plt
from random import randint
from Neural_Net import *
import time

class Train:
    def __init__(self, net):
        self.inodes = net.inodes
        self.hnodes = net.hnodes
        self.onodes = net.onodes
        self.lr = net.lr
        self.net = net
        self.performance_data = []

    def train(self, ep, pe, t, eff, est, epochs):
        training_file = open('antialiased_data.csv') #Gets training data
        training_list = training_file.readlines()
        training_file.close()
        start = time.time()
        final = 0
        performance_list = []
        for counter in range(epochs):
            epoch_start = time.time()
            for record in training_list:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(26) + 0.01
                targets[int(all_values[0])] = 0.99
                self.net.train(inputs, targets)
            scorecard = []
            test_data_file = open('antialiased_test_data.csv')
            test_data_list = test_data_file.readlines()
            test_data_file.close()
            for record in test_data_list:
                all_values2 = record.split(',')
                correct_label = int(all_values2[0])
                inputs = ((numpy.asfarray(all_values2[1:]) / 255.0 * 0.99) + 0.01)
                outputs = self.net.query(inputs)
                #finds highest one of arguments
                label = numpy.argmax(outputs)
                if (label == correct_label):
                    scorecard.append(1)
                else:
                    scorecard.append(0)
                    # calculate the performance score, the fraction of correct answers
            scorecard_array = numpy.asarray(scorecard)
            final = scorecard_array.sum() / scorecard_array.size * 100
            performance_list.insert(len(performance_list), final)
            epoch_end = time.time() - epoch_start
            remaining_time = int(epoch_end*(epochs-counter-1))
            minutes = remaining_time // 60
            seconds = remaining_time % 60
            ep.set('Epoch #{}'.format(counter+1))
            pe.set('Performance = {:.2f}%'.format(final))
            t.set('Time per epoch = {:.2f} seconds'.format(epoch_end))
            eff.set('Efficiency = {:.2f} entries/second'.format(len(training_list)/epoch_end))
            est.set('Time Remaining {} {}:{:02d}'.format(chr(126), minutes, seconds))
        end = (time.time() - start) / epochs
        self.performance_data = performance_list
