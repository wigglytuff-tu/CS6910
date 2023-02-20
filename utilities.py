import numpy as np

# Data Preprocessing
def normalize(x, eps=1e-5):
 min = np.min(x)
 max = np.max(x)
 return (x - min) / (max - min)
def onehot(x, num_classes = 10):
  a = np.zeros(10)
  a[x] = 1.
  return a

# Activation Functions
def sigmoid(x):
 return 1/(1+np.exp(-x))
def tanh(x):
 return np.tanh(x)
def relu(x):
 return np.where(x>0, x, np.zeros(1,))

# Output probabilities
def softmax(x):
 return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))

# Defining loss functions
def cross_entropy(y_pred, y_true):
 return np.mean(-(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred)))

# Define known gradients of loss function and activation functions
def grad_cross_entropy(y_pred, y_true):
 grad = y_pred
 grad[range(y_true.shape[0]), np.argmax(y_true, dim=1)] -= 1
 grad = grad/y_true.shape[0]
 return grad
def grad_sigmoid(z):
 return z*(1-z)
def grad_tanh(z):
 return 1-z*z
def grad_relu(z):
 return np.where(z>0, np.ones(1,), np.zeros(1,))

# Metrics
def get_accuracy(y_pred, y_true):
 return (np.argmax(y_pred, dim=1)==np.argmax(y_true, dim=1)).float().sum().item()

# Initialize weights and biases
class WeightsBiases():
 def __init__(self, in_channels, out_channels):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.weight = np.random.uniform(size=(out_channels, in_channels))
    self.bias = np.zeros(out_channels)
    self.delta = np.zeros(out_channels)

# Define Model class
class Model1():
 def __init__(self, layers, activation='sigmoid', regularization=False):
    network = []
    for i in range(len(layers)-1):
        network.append(WeightsBiases(layers[i], layers[i+1]))
        self.network = network
        if activation=='sigmoid':
            self.act = sigmoid
            self.grad = grad_sigmoid
        elif activation=='tanh':
            self.act = tanh
            self.grad = grad_tanh
        elif activation=='relu':
            self.act = relu
            self.grad = grad_relu
            self.alpha = float(regularization)
 def forward(self, x):
    out = x
    for i, layer in enumerate(self.network):
        if i!=len(self.network)-1:
          out = self.act(out@layer.weight.T + layer.bias)
        elif i==len(self.network)-1:
          out = softmax(out@layer.weight.T + layer.bias)
    return out

 def backward(self, x, y_true, lr):
    outs = [x]
    for i, layer in enumerate(self.network):
      if i!=len(self.network)-1:
        outs.append(self.act(outs[-1]@layer.weight.T + layer.bias))
      elif i==len(self.network)-1:
        outs.append(softmax(outs[-1]@layer.weight.T + layer.bias))
    self.network[-1].delta = grad_cross_entropy(outs[-1], y_true).T
    for i in range(len(self.network)-2, -1, -1):
      self.network[i].delta = (self.network[i+1].weight.T@self.network[i+1].delta) * self.grad(outs[i+1]).T
    for i, layer in enumerate(self.network):
      layer.weight -= lr * layer.delta@outs[i] + self.alpha * layer.weight
      layer.bias -= lr * np.sum(layer.delta, dim=1)




