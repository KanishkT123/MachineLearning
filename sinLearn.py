import numpy as np
import matplotlib.pyplot as plt

def dataGen(plot=True):
	"""Generates sin data for the MLP"""
	x = np.ones((1,40))*np.linspace(0,1,40)	
	t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40)*0.2
	
	#Transpose
	t = t.T
	x = x.T

	#Plot data
	if plot:
		plt.plot(x,t)
		plt.show()

	train = x[0::2,:]
	test = x[1::4,:]
	valid = x[3::4,:]
	traintarget = t[0::2,:]
	testtarget = t[1::4,:]
	validtarget = t[3::4,:]
dataGen()