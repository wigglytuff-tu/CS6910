import numpy as np
import pandas as pd
import argparse
from helper import *
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt


def config_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-wp','--wandb_project',default='CS6910',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we','--wandb_entity',default='purvam',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset',default='fashion_mnist', 
                        choices = ['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs',default=5,
                        help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size',default=32,
                        help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss',default='cross_entropy',
                        choices = ['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer',default='sgd',
                        choices = ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate',default=0.001,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum',default=0.9,
                        help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta',default=0.9,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1',default=0.9,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2',default=0.999,
                        help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', default=0.0000001,
                        help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay',default=0.0,
                        help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', default='random',
                        choices = ['random', 'Xavier'])
    parser.add_argument('-nhl', '--num_layers',default=3,
                        help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size',default=32,
                        help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation',default='ReLU',
                        choices = ['identity', 'sigmoid', 'tanh', 'ReLU'])
    return parser

parser = config_parser()
args = parser.parse_args()
  
def get_activation(name):
    if name == 'ReLU':
        return RelU()
    elif name == 'tanh':
        return Tanh()
    elif name == 'sigmoid':
        return Sigmoid()

def get_optimizer(name, lr):
    if name == 'sgd':
        return SGD(lr=lr)
    elif name == 'momentum':
        return Momentum(lr=lr)
    elif name == 'nag':
        return NAG(lr=lr)
    elif name == 'rmsprop':
        return RMSprop(lr=lr)
    elif name == 'adam':
        return Adam(lr=lr)
    elif name == 'nadam':
        return Nadam(lr=lr)

def get_loss(name):
    if name == 'cross_entropy':
        return CrossEntropyLossFromLogits()
    elif name == 'mean_squared_error':
        return MSELossFromLogits()

def create_layers(n_layers, layer_size, activation, weight_decay, init_method):

    layers = []
    layers.extend([FC(784,layer_size, weight_decay, init_method), get_activation(activation)])
    
    for _ in range(n_layers):
        layers.extend([FC(layer_size, layer_size, weight_decay, init_method), get_activation(activation)])
    
    layers.append(FC(layer_size, 10, weight_decay, init_method))

    return layers
#Function used for hyperparamter tuning using wandb sweeps

def train():

  wandb.init(project=args.wandb_project, entity=args.wandb_entity, magic=True)

  X_train, y_train, X_test, y_test, X_val, y_val = dataloader(args.dataset)

  model = NeuralNet(create_layers(args.num_layers, 
                                  args.hidden_size, 
                                  args.activation, 
                                  args.weight_decay,
                                  args.weight_init),use_wandb=True)


  model.compile(loss=get_loss(args.loss), optimizer=get_optimizer(args.optimizer, args.learning_rate))

  model.fit(X_train, y_train, X_val, y_val, batch_size=args.batch_size, epochs=args.epochs)

def get_confusion_matrix():

    config_defaults = {
        'n_layers': 4,
        'layer_size': 128,
        'weight_decay': 0,
        'lr': 1e-3,
        'loss': 'cross_entropy',
        'optimizer': 'adam',
        'batch_size': 128,
        'init_method': 'xavier',
        'activation': 'relu',
        'epochs': 10
    }

    wandb.init(project=args.wandb_project , entity=args.wandb_entity, config=config_defaults, magic=True)

    X_train, y_train, X_test, y_test, X_val, y_val = dataloader(args.d)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

     

    model = NeuralNet(create_layers(wandb.config.n_layers, 
                                    wandb.config.layer_size, 
                                    wandb.config.activation, 
                                    wandb.config.weight_decay,
                                    wandb.config.init_method))

    
    model.compile(loss=get_loss(wandb.config.loss), optimizer=get_optimizer(wandb.config.optimizer, wandb.config.lr))
    model.fit(X_train, y_train, X_test, y_test, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs)

    preds = np.argmax(model(X_test), axis=1)
    wandb.log({'conf_mat' : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_test, preds=preds,
                        class_names=class_names)})

def get_images():
      
    wandb.init(project='DL', entity='purvam', name='log_images')
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    indices = [list(y_train.T).index(i) for i in range(10)]
    images = []
    captions = []
    for index in indices:
      images.append(X_train[index].reshape((28, 28)))
      captions.append(class_names[y_train[index]])
    wandb.log({'Image from each class': [wandb.Image(image, caption=caption) for image, caption in zip(images, captions)]})
 
train()
