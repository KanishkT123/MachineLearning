#A basic Multilayer perceptron
#Made just for practice 

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
		
