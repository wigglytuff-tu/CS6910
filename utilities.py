import numpy as np

# Data Preprocessing techniques

def normalize(x, eps=1e-5):
 min = np.min(x)
 max = np.max(x)
 return (x - min) / (max - min)

# Activation Functions
def sigmoid(x):
 return 1/(1+np.exp(-x))
def softmax(x):
 return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
def tanh(x):
 return np.tanh(x)
def relu(x):
 return np.where(x>0, x, np.zeros(1,))