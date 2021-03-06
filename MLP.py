#A basic Multilayer perceptron
#Made just for practice

import math
import random
import numpy as np
import copy

#Activation functions and their derivatives

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

#hyperbolic tangent function
def tanh(x):
    return math.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y*y



class MLP(object):

	def __init__(self, inputN, hiddenL, outputN, learning):

		"""
		inputN: Input neurons
		hidennL: Number of neurons in each hidden layer (as a List)
		outputN: output layer neurons
		"""
		self.inputN = inputN + 1 #Bias neuron
		self.outputN = outputN
		self.hiddenL = hiddenL

		#Activation list
		self.ai = np.ones(self.inputN)
		self.ah = np.asarray([np.ones(nodes) for nodes in self.hiddenL])
		self.ao = np.ones(self.outputN)

		#Random weights for input to hidden, hidden to hidden, hidden to output
		self.wi = np.random.randn(self.inputN, self.hiddenL[0])
		self.wh = np.asarray([np.random.randn(self.hiddenL[x], self.hiddenL[x+1]) for x in range(len(self.hiddenL)-1)])
		self.wo = np.random.randn(self.hiddenL[-1], self.outputN)

		#Arrays for weight changes
		self.ci = np.zeros((self.inputN, self.hiddenL[0]))
		self.ch = np.asarray([np.zeros((self.hiddenL[x], self.hiddenL[x+1])) for x in range(len(self.hiddenL)-1)])
		self.co - np.zeros((self.hiddenL[-1], self.outputN))

		#array for storing output deltas
		#dummy for input, then hidden, then output all as numpy arrays
		self.sensitivity = np.asarray([np.asarray([])] + [np.zeros(layer) for layer in self.hiddenL] + [np.zeros(outputN)])

		#learning rate
		self.learning = learning

	def feedForward(self, inputs):
		if len(inputs) != self.inputN -1:
			raise ValueError("Input has wrong length: " + str(len(inputs)) + 
			" as compared to: " + str(self.inputN - 1))

		#Activations for the input layer
		for i in range(self.inputN -1):
			self.ai[i] = inputs[i]

		#Activations for the first hidden layer (done separately to make loops easier)
		#calculate the activation in the hidden layer by going through activation function
		self.ah = sigmoid(np.dot(self.ai, self.wi))
		
		#Activations for non hidden layer
		#For every hidden layer after the first
		for layer in range(1, len(self.hiddenL)):
			self.ah[layer] = sigmoid(np.dot(self.ah[layer-1], self.wh[layer-1]))

		#activations for output layer
		self.ao = sigmoid(np.dot(self.ah[-1], self.wo))

		return self.ao

	def errorCalc(self, targets):
		'''
		Calculates overall error in the MLP given the target vector
		'''
		if len(targets) != self.outputN:
			raise ValueError("Given target length: " + str(len(targets))
				+ " Expected: " + str(self.outputN))

		error = targets - self.ao
		error = error * error
		error = error * 0.5
		return sum(error)

	def backPropagate(self, targets):
		"""
		targets: the y values to be used for learning
		returns updated weights and the current error
		"""

		if len(targets) != self.outputN:
			raise ValueError("Given target length: " + str(len(targets))
				+ " Expected: " + str(self.outputN))
		 	
		#Calculating the direction in which the output needs to change
		#-(target-out) = out - target
		errorAgainstOutput = self.ao - targets
		self.sensitivity[-1] = errorAgainstOutput * dsigmoid(self.ao)
				

		#Iterating backwards over all the hidden layers
		for layer in range(len(hiddenL), 0, -1):
			for neuron in range(hiddenL[layer]):
				factor = dsigmoid(self.ah[layer][neuron])
				sum = 0
				#Different rules if it's the last hidden layer or not
				if layer == len(hiddenL) - 1:
					for nextNeuron in range(outputN):
						sum += self.wo[neuron][nextNeuron]*self.sensitivity[-1][nextNeuron]
				else:
					for nextNeuron in range(hiddenL[layer+1]):
						sum += self.wh[neuron][nextNeuron]*self.sensitivity[layer+1][nextNeuron]
				self.sensitivity[layer] = sum*factor

	def update(self):
















