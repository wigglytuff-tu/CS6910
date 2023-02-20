import numpy as np
import pandas as pd
import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c','--config', is_config_file=True, help='config file path')
    parser.add_argument('-wp','--wandb_project',default='CS6910',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we','wandb_entity',default='purvam',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset',default='fashion_mnist',
                        help='choices: ["mnist", "fashion_mnist"]')
    parser.add_argument('-e', '--epochs',default=1,
                        help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size',default=4,
                        help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss',default='cross_entropy',
                        help='choices: ["mean_squared_error", "cross_entropy"]')
    parser.add_argument('-o', '--optimizer',default='sgd',
                        help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-lr', '--learning_rate',default=0.1,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum',default=0.5,
                        help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta',default=0.5,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1',default=0.5,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2',default=0.5,
                        help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', default=0.000001,
                        help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay',default=0.0,
                        help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', default='random',
                        help='choices: ["random", "Xavier"]')
    parser.add_argument('-nhl', '--num_layers',default=1,
                        help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size',default=4,
                        help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation',default='sigmoid',
                        help='choices: ["identity", "sigmoid", "tanh", "ReLU"]')
    return parser


