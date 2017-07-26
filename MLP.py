#A basic Multilayer perceptron
#Made just for practice

#Activation functions and their derivatives

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)    


class MLP(object):

	def __init__(self, inputN, hiddenL, outputN):

		"""
		inputN: Input neurons
		hidennL: Number of neurons in each hidden layer (as a List)
		outputN: output layer neurons
		"""
		self.inputN = inputN + 1 #Bias neuron
		self.outputN = outputN
		self.hiddenL = hiddenL

		#Activation list
		self.ai = [1.0] * self.inputN
		self.ao = [1.0] * self.outputN
		self.ah = [[1.0]*nodes for nodes in self.hiddenL]

		#Random weights
		self.wi = np.random.randn(self.inputN, self.hiddenL[0])
		self.wh = [np.random.randn(self.hiddenL[x], self.hiddenL[x+1]) for x in range(len(self.hiddenL))]
		self.wo = np.random.randn(self.hiddenL[-1], self.outputN)

		#Arrays for weight changes
		self.ci = np.zeros((self.inputN, self.hiddenL[0]))
		self.ch = [np.zeros((self.hiddenL[x], self.hiddenL[x+1])) for x in range(len(self.hiddenL))]
		self.co - np.zeros((self.hiddenL[-1], self.outputN))


	def feedForward(self, inputs):
		if len(inputs) != self.inputN -1:
			raise ValueError("Input has wrong length: " + str(len(inputs)))
		#Activations for the input layer
		for i in range(self.inputN -1):
			self.ai[i] = inputs[i]

		#Activations for the first hidden layer (done separately to make loops easier)
		for neuron in range(self.hiddenL[0]):
			total = 0.0
			for prevNeuron in range(self.inputN):
				total += self.ai[prevNeuron]* self.wi[prevNeuron][neuron]
			#calculate the activation in the hidden layer by going through activation function
			self.ah[neuron] = sigmoid(total)
		
		#Activations for non hidden layer
		#For every hidden layer after the first
		for layer in range(1, len(self.hiddenL)):
			#for every neuron in that hidden layer
			for neuron in range(self.hiddenL[layer]):
				total = 0.0
				#total up the activations from the neurons of the previous layer times the connection weight
				for prevNeuron in range(self.hiddenL[layer-1]):
					total += self.ah[layer-1][prevNeuron] * self.wh[layer-1][prevNeuron][neuron]
				self.ah[layer][neuron] = sigmoid(total)

		#activations for output layer
		for neuron in range(self.outputN):
			total = 0.0
			for prevNeuron in range(self.hiddenL[-1]):
				total += self.ah[-1][prevNeuron] * self.wo[prevNeuron][neuron]
			self.ao[neuron] = sigmoid(total)

		return self.ao[:]


















