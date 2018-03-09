# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 08:58:27 2018

@author: Matty Boy
"""
from scipy.optimize import minimize
import numpy as np
from scipy.io import loadmat
from math import sqrt
from functools import reduce



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W
# Paste your sigmoid function here

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    #Plugin the sigmoid function from lecture
    #1/(1+e^(-net) where net is input z
    #np.exp is python e operator
    return  1/(1 + np.exp(-1 * z))

# Paste your nnObjFunction here
def nnObjFunction(params, *args):
    
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    a = np.array([])
    z = np.array([])
    b = np.array([])
    o = np.array([])
    JW1W2 = 0
    JW1 = np.array([])
    JW2 = np.array([])
    lim = len(training_data[:,0]);
    for i in range(0, lim):
        ai = np.array([])
        zi = np.array([])
        bias_td = np.append(training_data[i], [1])
        # formula 1
      
        ai = np.append(np.matmul(w1, np.transpose(bias_td)), ai)
        # formula 2
        zi = np.append(sigmoid(ai), zi)
            
        try:
            a = np.vstack((a, ai))
        except:
            a = ai
            
        try:
            z = np.vstack((z, zi))
        except:
            z = zi
            
        bi = np.array([])
        oi = np.array([])
        ai = np.append(ai, [1])
        # formula 3
        
        bi = np.append(np.matmul(w2, np.transpose(ai)), bi)
        # formula 4
        oi = np.append(sigmoid(bi), oi)
            
        try:
            b = np.vstack((b, bi))
        except:
            b = bi
            
        try:
            o = np.vstack((o, oi))
        except:
            o = oi
        
        JW1W2i = 0
        for l in range(0, n_class):
            JW1W2i -= (training_label[l] * np.log(oi[l]) + (1 - training_label[l]) * np.log(1 - oi[l]))
        Si = np.subtract(oi, training_label);
        dJiW2 = np.matmul(np.transpose(Si[np.newaxis]), zi[np.newaxis])
        JW1W2 += np.sum((1/len(training_data[:,0])) * JW1W2i)
        t_dJiW2 = w2
        t_dJiW2[:,:-1] = dJiW2
        try:
            JW2 = np.add(t_dJiW2, JW2)
        except:
            JW2 = t_dJiW2
        
        dJiW1p1 = np.matmul(1-zi, zi)
        dJiW1p2 = np.sum(np.matmul(Si, w2))
        dJiW1 = np.matmul(((dJiW1p1 * dJiW1p2)[np.newaxis]),(training_data[i,:][np.newaxis]))
        t_dJiW1 = w1
        t_dJiW1[:,:-1] = dJiW1
        try:
            JW1 = np.add(t_dJiW1, JW1)
        except:
            JW1 = t_dJiW1
        
    JW1 /= lim
    JW2 /= lim
    JW1W2 /= lim
    obj_val = JW1W2 + (lambdaval / (2 * lim)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((JW1.flatten(), JW2.flatten()),0)
    

    return (obj_val, obj_grad)

n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)





#minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
objval,objgrad = nnObjFunction(params, *args)

print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)
