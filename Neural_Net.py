import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        #Set number of nodes in input, hidden, and output later
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #Link weight matrices
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #Learning rate
        self.lr = learningrate

        #Sigma activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):

        #Convert input lists to 2d arrays
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #Calculate signals into hidden layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #Calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #Update weights for links between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        #Update weights for links between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        #Convert input list to 2d array
        self.inputs = numpy.array(inputs_list, ndmin=2).T

        #Calculate signals into hidden layer
        self.hidden_inputs = numpy.dot(self.wih, self.inputs)
        #Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(self.hidden_inputs)

        #Calculate signals into final output layer
        self.final_inputs = numpy.dot(self.who, hidden_outputs)
        #Calculate signals emerging from final layer
        final_outputs = self.activation_function(self.final_inputs)

        return final_outputs
