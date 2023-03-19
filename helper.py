from template import *
import numpy as np
from copy import deepcopy
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist

# Data Preprocessing
def normalize(x, eps=1e-5):
 # Normalize the input data x to bring it in the range of [0.,1.]
 min = np.min(x)
 max = np.max(x)
 return (x - min) / (max - min)
def flatten(x):
    x = x.reshape(x.shape[0], -1)
    return x


# Data Loader
def dataloader(dataset_name : str) -> np.array:
  if dataset_name == "mnist":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
  elif dataset_name == "fashion_mnist":
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  # Preprocess the data -- Normalize  
  X_train , X_test = normalize(X_train) , normalize(X_test)
  # Segregate into validation set(10% of train set) and training set
  idx = np.random.choice(range(len(y_train)),size=len(y_train)//10,replace=False)
  y_val = y_train[idx]
  X_val = X_train[idx]
  X_train = np.delete(X_train,idx,axis=0)
  y_train = np.delete(y_train,idx)
  X_train, X_test, X_val = flatten(X_train), flatten(X_test), flatten(X_val)
  return X_train, y_train, X_test, y_test, X_val, y_val

#Activation Functions

class Sigmoid(AutoDiffFunction):
    """ 
    Represents the Sigmoid Activation function
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        self.saved_for_backward = 1/(1 + np.exp(-x))
        return self.saved_for_backward

    def compute_grad(self, x):
        y = self.saved_for_backward

        return {"x": y*(1-y)}

    def backward(self, dy):
        return dy * self.grad["x"]      


class RelU(AutoDiffFunction):
    """ 
    Represents the RelU Activation function
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        self.saved_for_backward = np.where(x>0.0, 1.0, 0.0)

        return x * self.saved_for_backward

    def compute_grad(self, x):
        return {"x": self.saved_for_backward}

    def backward(self, dy):
        return dy * self.grad["x"]
    
class Tanh(AutoDiffFunction):
    """ 
    Represents the Tanh Activation function
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        self.saved_for_backward = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        return self.saved_for_backward

    def compute_grad(self, x):
        y = self.saved_for_backward

        return {"x": 1 - y**2}

    def backward(self, dy):
        return dy * self.grad["x"]
    
# Layer

class FC(Layer):
    def __init__(self, in_dim, out_dim, weight_decay=None, init_method="random") -> None:
        super().__init__()
        self.weight_decay = weight_decay
        self.init_method = init_method
        self.initialize_weights(in_dim, out_dim)

    def initialize_weights(self, in_dim, out_dim):
        
        if self.init_method == "random":
            scaling_factor = 1/np.sqrt(in_dim)
            self.weights["w"] = np.random.randn(in_dim, out_dim) * scaling_factor
            self.weights["b"] = np.random.randn(1, out_dim) * scaling_factor
        elif self.init_method == "xavier":
            lim = np.sqrt(6 / (in_dim + out_dim))
            self.weights["w"] = np.random.uniform(low=-lim, high=lim, size=(in_dim, out_dim))
            self.weights["b"] = np.random.uniform(low=-lim, high=lim, size=(1, out_dim))

    def compute_grad(self, x):
        
        gradients = {}

        # y = x * w + b        
        # we compute gradients wrt w and x 
        # gradient wrt b is not required explicitly since we know that it's value is 1
        gradients["w"] = self.saved_for_backward["x"].T
        gradients["x"] = self.weights["w"].T

        return gradients


    def forward(self, x):
        output = x @ self.weights["w"] + self.weights["b"]
        self.saved_for_backward["x"] = x
        
        return output

    def backward(self, dy):
        
        # calculating gradients wrt input to pass on to previous layer for backprop
        dx = dy @ self.grad["x"]
        
        # calculating gradients wrt weights
        dw = self.grad["w"] @ dy
        db = np.sum(dy, axis=0, keepdims=True)

        # accomodating for weight_decay / L2 regularization
        if self.weight_decay:
            dw = dw + 2 * self.weight_decay * self.weights["w"]
            db = db + 2 * self.weight_decay * self.weights["b"]

        self.absolute_gradients = {"w": dw, "b": db}

        return dx

    def update_weights(self):
        self.optimizer.step(self)

# 1. CROSSENTROPY LOSS
class CrossEntropyLossFromLogits(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.n_classes = None

    @staticmethod
    def softmax(x):
        v = np.exp(x)
        return v / np.sum(v, axis=1, keepdims=True)

    def encode(self, y): 
        encoded_y = np.zeros(shape=(len(y), self.n_classes))

        for i in range(len(y)):
            encoded_y[i,y[i]] = 1

        return encoded_y

    def forward(self, y_pred, y_true):
         
        probabilities = self.softmax(y_pred)
        y_true_encoded = self.encode(y_true)

        loss_value = np.mean(np.sum(- y_true_encoded * np.log(probabilities), axis=1))

        self.saved_for_backward["probabilities"] = probabilities
        self.saved_for_backward["y_true"] = y_true_encoded

        return loss_value

    def compute_grad(self, y_pred, y_true):

        return {"x": self.saved_for_backward["probabilities"] - self.saved_for_backward["y_true"]}        


# 2. MEAN SQUARED LOSS
class MSELossFromLogits(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.n_classes = None

    @staticmethod
    def softmax(x):
        v = np.exp(x)

        return v / np.sum(v, axis=1, keepdims=True)

    def encode(self, y): 
        encoded_y = np.zeros(shape=(len(y), self.n_classes))

        for i in range(len(y)):
            encoded_y[i,y[i]] = 1

        return encoded_y
    
    @staticmethod
    def indicator(i, j):
        ind = {True: 1, False: 0}
        return ind[i==j]

    def forward(self, y_pred, y_true):
         
        probabilities = self.softmax(y_pred)
        y_true_encoded = self.encode(y_true)

        loss_value = np.mean(np.sum((probabilities - y_true_encoded)**2, axis=1))

        self.saved_for_backward["probabilities"] = probabilities
        self.saved_for_backward["y_true"] = y_true_encoded

        return loss_value

    def compute_grad(self, y_pred, y_true):

        probs = self.saved_for_backward["probabilities"]
        labels = self.saved_for_backward["y_true"]
        grad = np.zeros(shape=(len(y_true), self.n_classes))
        
        for point_counter in range(len(y_true)):
            res = 0
            for i in range(self.n_classes):
                for j in range(self.n_classes):
                    
                    res = probs[point_counter, j] * (probs[point_counter, j] - labels[point_counter, j]) * (self.indicator(i,j) - probs[point_counter, i])
                
                grad[point_counter, i] = res
        
        return {"x": grad}

# 1. SGD OPTIMIZER
class SGD(Optimizer):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.lr = lr

    def step(self, layer):

        for weight_name, _ in layer.weights.items():
            layer.weights[weight_name] = layer.weights[weight_name] - self.lr * layer.absolute_gradients[weight_name]
            

# 2. MOMENTUM OPTIMIZER
class Momentum(Optimizer):
    def __init__(self, lr=1e-3, gamma=0.9):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        
    def step(self, layer):
        
        #Initialise update history
        if self.remember == {}:
            for weight_name, weight in layer.weights.items():
                self.remember[weight_name] = {}
                self.remember[weight_name]["v"] = np.zeros_like(weight)
        
        #Momentum update rule
        for weight_name, weight in layer.weights.items():
            self.remember[weight_name]["v"] = self.gamma * self.remember[weight_name]["v"] + \
                                                self.lr * layer.absolute_gradients[weight_name]
            layer.weights[weight_name] = layer.weights[weight_name] - self.remember[weight_name]["v"]

            
# 3. NESTEROV ACCELARATED GRADIENT OPTIMIZER
class NAG(Optimizer):
    def __init__(self, lr=1e-3, gamma=0.9):
        super().__init__()
        self.lr = lr
        self.gamma = gamma 

    def step(self, layer):

        if self.remember == {}:
            for weight_name, weight in layer.weights.items():
                self.remember[weight_name] = {}
                self.remember[weight_name]["v"] = np.zeros_like(weight)

        for weight_name, weight in layer.weights.items():
            layer.weights[weight_name] = layer.weights[weight_name] + (self.gamma**2) * self.remember[weight_name]["v"] - \
                                            (1 + self.gamma) * self.lr * layer.absolute_gradients[weight_name]

            self.remember[weight_name]["v"] = self.remember[weight_name]["v"] * self.gamma - \
                                                self.lr * layer.absolute_gradients[weight_name]

# 4. RMSPROP OPTIMIZER
class RMSprop(Optimizer):
    def __init__(self, lr=1e-3, beta=0.9, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        
    def step(self, layer):
        
        #Initialise update history
        if self.remember == {}:
            for weight_name, weight in layer.weights.items():
                self.remember[weight_name] = {}
                self.remember[weight_name]["v"] = np.zeros_like(weight)
        
        #RMSprop update rule
        for weight_name, weight in layer.weights.items():
            self.remember[weight_name]["v"] = self.beta * self.remember[weight_name]["v"] + \
                                                (1 - self.beta) * (layer.absolute_gradients[weight_name] ** 2)
            layer.weights[weight_name] = layer.weights[weight_name] - (self.lr / (np.sqrt(self.remember[weight_name]["v"] + \
                                                self.epsilon))) * layer.weights[weight_name]

            
# 5. ADAM OPTIMIZER
class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1
        
    def step(self, layer):
        
        #Initialise update history
        if self.remember == {}:
            for weight_name, weight in layer.weights.items():
                self.remember[weight_name] = {}
                self.remember[weight_name]["v"] = np.zeros_like(weight)
                self.remember[weight_name]["m"] = np.zeros_like(weight)
        
        #Adam update rule
        for weight_name, weight in layer.weights.items():
            
            #Update m_t and v_t
            self.remember[weight_name]["m"] = self.beta_1 * self.remember[weight_name]["m"] + \
                                                (1 -self.beta_1) * layer.absolute_gradients[weight_name]
            
            self.remember[weight_name]["v"] = self.beta_2 * self.remember[weight_name]["v"] + \
                                                (1 - self.beta_2) * (layer.absolute_gradients[weight_name]**2)
            
            #Bias correction
            m_hat = self.remember[weight_name]["m"]/(1 - self.beta_1 ** self.t)
            v_hat = self.remember[weight_name]["v"]/(1 - self.beta_2 ** self.t)
            
            #Update parameters
            layer.weights[weight_name] = layer.weights[weight_name] - (self.lr / (np.sqrt(v_hat + self.epsilon))) * m_hat
            
        self.t += 1
            
            
# 6. NADAM OPTIMIZER            
class Nadam(Optimizer):
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__()
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1

    def step(self, layer):
        
        # we have 2 parameters to remember m(t) and v(t) for all weights in the layer
        if self.remember == {}:
            for weight_name, weight in layer.weights.items():
                self.remember[weight_name] = {}
                self.remember[weight_name]["v"] = np.zeros_like(weight)
                self.remember[weight_name]["m"] = np.zeros_like(weight)

        for weight_name, weight in layer.weights.items():
            
            self.remember[weight_name]["m"] = self.beta_1 * self.remember[weight_name]["m"] + \
                                                (1 -self.beta_1) * layer.absolute_gradients[weight_name]

            self.remember[weight_name]["v"] = self.beta_2 * self.remember[weight_name]["v"] + \
                                                (1 - self.beta_2) * layer.absolute_gradients[weight_name]**2

            # bias correction step 
            m_hat = self.remember[weight_name]["m"]/(1 - self.beta_1 ** self.t)
            v_hat = self.remember[weight_name]["v"]/(1 - self.beta_2 ** self.t)

            d = self.lr / (np.sqrt(v_hat) + self.epsilon) * (self.beta_1*m_hat + (1-self.beta_1)/
                                                (1-self.beta_1 ** self.t) * layer.absolute_gradients[weight_name]) 

            layer.weights[weight_name] = layer.weights[weight_name] - d

        self.t += 1

class NeuralNet():
    def __init__(self, layers, use_wandb=True) -> None:
        self.layers = layers
        self.use_wandb = use_wandb
        self.history = []

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def compile(self, loss, optimizer):
        self.loss = loss

        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.optimizer = deepcopy(optimizer) # each layer has it's own optimizer

    def forward(self, x):
        """Performs forward pass for the entire network

        Args:
            x (np.ndarray): input array

        Returns:
            np.ndarray: output of the neural network
        """
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self):
        """Performs one backward pass for the entire network
            and saves the gradients

        Returns:
            np.ndarray: gradient of the 1st layer
        """

        gradient = self.loss.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def update_weights(self):
        """Updates weights of all layers using the chosen optimizer 
            and saved gradients
        """

        for layer in reversed(self.layers):
            if isinstance(layer, Layer):
                layer.update_weights()

    @staticmethod
    def accuracy_score(y_pred, y_true):
        """Returns accuracy score

        Args:
            y_pred (np.ndarray): predicted labels (batch_size X n_classes)
            y_true (np.ndarray): true labels (batch_size X 1)

        Returns:
            [float]: accuracy in fraction
        """

        pred_labels = np.argmax(y_pred, axis=1)
        return np.sum(pred_labels == y_true) / len(y_true)

    @staticmethod
    def create_batches(X, y, batch_size=32):
        """Creates batches from given dataset of given size

        Args:
            X (np.ndarray): input features
            y (np.ndarray): labels
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            list: batches of data as list of (x,y) tuples
        """
        batches = []

        for i in range(len(y) // batch_size):
            start_idx = batch_size * i
            end_idx = batch_size * (i + 1)

            batches.append([X[start_idx: end_idx], y[start_idx: end_idx]])

        # take care of the last batch which might have batch_size less than the specified one
        if len(y) % batch_size != 0:
            batches.append([X[end_idx:], y[end_idx:]])

        return batches

    def fit(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
        """Fits the model onto the given data

        Args:
            X_train (np.ndarray): train features
            y_train (np.ndarray): train labels
            X_val (np.ndarray): validation features
            y_val (np.ndarray): validation labels
            batch_size (int, optional): batch size. Defaults to 32.
            epochs (int, optional): number of epochs to train the model. Defaults to 10.
        """

        # calculate number of classes to pass to the loss function
        self.loss.n_classes = len(np.unique(y_train))

        train_batches = self.create_batches(X_train, y_train, batch_size=batch_size)
        val_batches = self.create_batches(X_val, y_val, batch_size=batch_size)

        num_train_batches = len(train_batches)
        num_val_batches = len(val_batches)

        for epoch in range(1, epochs+1):

            total_train_loss = 0
            total_train_accuracy = 0

            ## TRAINING ##
            for X, y in train_batches:

                preds = self(X)
                total_train_loss += self.loss(preds, y)
                total_train_accuracy += self.accuracy_score(preds, y)

                _ = self.backward()
                self.update_weights()

            train_loss_per_epoch = total_train_loss / num_train_batches
            train_accuracy = total_train_accuracy / num_train_batches

            total_val_loss = 0
            total_val_accuracy = 0

            ## VALIDATION ##
            for X_v, y_v in val_batches:
                val_preds = self(X_v)
                total_val_loss += self.loss(val_preds, y_v)
                total_val_accuracy += self.accuracy_score(val_preds, y_v)
            
            val_loss_per_epoch = total_val_loss / num_val_batches
            val_accuracy = total_val_accuracy / num_val_batches
            
            print(f"Epoch: {epoch} Train Loss: {train_loss_per_epoch} Train Accuracy: {train_accuracy} Val Loss: {val_loss_per_epoch} Val Accuracy: {val_accuracy}")

            self.history.append({"Epoch" : epoch, 
                                    "Train Loss": train_loss_per_epoch,
                                    "Train Accuracy": train_accuracy,
                                    "Val Loss": val_loss_per_epoch,
                                    "Val Accuracy": val_accuracy})
            if self.use_wandb:
                wandb.log(self.history[-1])

        print("\nModel trained successfully!")

    def evaluate(self, X_test, y_test):
        """Evaluates the model on a test dataset

        Args:
            X_test (np.ndarray): test features 
            y_test (np.ndarray): test labels
        """

        preds = self(X_test)
        test_loss = self.loss(preds, y_test)
        accuracy = self.accuracy_score(preds, y_test)

        print(f"Test loss: {test_loss} Test accuracy: {accuracy}")