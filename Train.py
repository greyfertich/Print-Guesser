from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import csv
import matplotlib.pyplot
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

        # new_wih = self.net.wih.T
        # tot = np.zeros(self.inodes)
        # r = np.zeros((40,40))
        # g = np.zeros((40,40))
        # b = np.zeros((40,40))
        # high = 0
        # low = 0
        # for i in range(self.inodes):
        #     for c in range(self.hnodes):
        #         tot[i] += new_wih[i][c]
        #     if tot[i] > high: high = tot[i]
        #     if tot[i] < low: low = tot[i]
        #
        # for i in range(self.inodes):
        #     if tot[i] > 0: g[i//40][i%40] = tot[i] / high
        #     if tot[i] < 0: r[i//40][i%40] = tot[i] / low
        #
        # rgbArray = np.zeros((40,40,3), 'uint8')
        # rgbArray[..., 0] = r*256
        # rgbArray[..., 1] = g*256
        # rgbArray[..., 2] = b*256
        # img = Image.fromarray(rgbArray)
        # new_img = img.resize((400,400))
        # new_img.save("train_img.gif")
        # img2 = PhotoImage(file='train_img.gif')
        # lb = Label(master, image=img2, bd=0)
        # lb.image = img2
        # lb.pack(side=RIGHT)

    def train(self, ep, pe, t, eff, est, epochs):
        training_file = open('img_data.csv') #Gets training data
        training_list = training_file.readlines()
        training_file.close()
        num = 0
        start = time.time()
        final = 0
        for counter in range(epochs):
            epoch_start = time.time()
            num += 1
            c = 0
            for record in training_list:
                c += 1
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(26) + 0.01
                targets[int(all_values[0])] = 0.99
                self.net.train(inputs, targets)
            scorecard = []
            test_data_file = open('test_data.csv')
            test_data_list = test_data_file.readlines()
            test_data_file.close()
            for record in test_data_list:
                #if num % 5 == 0:
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
            epoch_end = time.time() - epoch_start
            ep.set('Epoch #{}'.format(counter+1))
            pe.set('Performance = {:.2f}%'.format(final))
            t.set('Time per epoch = {:.2f} seconds'.format(epoch_end))
            eff.set('Efficiency = {:.2f} entries/second'.format(len(training_list)/epoch_end))
            est.set('Time Remaining = {:.2f} seconds'.format((epochs-counter-1)*epoch_end))
        end = (time.time() - start) / epochs
