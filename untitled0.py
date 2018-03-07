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
opt_count = 0


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
    global opt_count
    opt_count+=1;
    print(opt_count)
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
    #print(w1)
    #print(w2)
    obj_val = 0
    eta = 0.1
    
    
    
    
    #use this line for objective func
    output_matrix = np.array([])
    h_matrix = np.array([])
    """isFirst= True
    for j in range(0,n_input):
        output_per_input = np.array([])
        
        
        bias_td = np.append(training_data[j], [1])
        h = np.array([])
        for i in w1:
            a = np.matmul(np.transpose(i), bias_td)
            z = sigmoid(a)
            h = np.append(h, z)
            
        h = np.append(h, 1)
        
        for i in w2:
            a = np.matmul(np.transpose(i), h)
            z = sigmoid(a)
            output_per_input = np.append(output_per_input, z)
            
        try:
            h_matrix = np.vstack((h_matrix,h))
        except:
            h_matrix = h
            
        if(isFirst):
            output_matrix = output_per_input
            isFirst = False;
        else:
            output_matrix = np.vstack((output_matrix,output_per_input))
       
    """
    a = np.array([])
    z = np.array([])
    b = np.array([])
    o = np.array([])
    JW1W2 = 0
    JW1 = np.ndarray([])
    JW2 = np.ndarray([])
    #lim = len(training_data[:,0]);
    lim = 1;
    for i in range(0, lim):
        ai = np.array([])
        zi = np.array([])
        bias_td = np.append(training_data[i], [1])
        # formula 1
        #w1[:,-1:] = [[1],[1],[1]]
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
        w2[:,-1:] = [[1],[1]]
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
        JW1W2 += np.sum((1/len(training_data[:,0])) * JW1W2i)
        
        dJiW2 = np.matmul(np.transpose(np.subtract(oi, training_label)[np.newaxis]), (zi)[np.newaxis])
        t_dJiW2 = w2
        t_dJiW2[:,:-1] = dJiW2
        try:
            JW2 = JW2.sum(t_dJiW2, JW2)
        except:
            JW2 = t_dJiW2
        
        dJiW1p1 = np.matmul(1-zi, zi)
        dJiW1p2 = np.sum(np.subtract(np.matmul(oi, w2), np.matmul(training_label, w2)))
        dJiW1 = ((dJiW1p1 * dJiW1p2)[np.newaxis]).dot(training_data[i,:][np.newaxis])
        t_dJiW1 = w1
        t_dJiW1[:,:-1] = dJiW1
        try:
            JW1 = JW1.sum(t_dJiW1, JW1)
        except:
            JW1 = t_dJiW1
        
        #print("old w2: " + np.array2string(w2))
        #w2 = np.subtract(w2, eta * (t_dJiW2 + (lambdaval * w2)))
        #print("new w2: " + np.array2string(w2))
        #print("old w1: " + np.array2string(w1))
        #w1 = np.subtract(w1, eta * (t_dJiW1 + (lambdaval * w1)))
        #print("new w1: " + np.array2string(w1))
        
        #dJiW1 = np.matmul(np.matmul(np.matmul(np.subtract(zi, 1), zi), np.matmul(np.subtract(oi, training_label), w2[i])), training_data[i])

    #w1 = np.subtract(w1, eta * JW1)
    w2 = np.subtract(w2, eta * JW2)

    JW1W2 /= len(training_data[:,0])
    #print("a: " + np.array2string(a))
    #print("z: " + np.array2string(z))
    #print("b: " + np.array2string(b))
    #print("o: " + np.array2string(o))
    obj_val = JW1W2 + (lambdaval / (2 * len(training_data[:, 0]))) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    # I think this is average error or gradiance, those are the same i think
    # whatever formula 5 is
    """sum_of_something = 0
    for i in range(0, len(training_data[:,0])):
        for l in range(0, n_class):
            sum_of_something += (training_label[i] * np.log(o[l]) + (1 - training_label[i]) * np.log(1 - o[l]))
    sum_of_something /= -1 * n_input
    sum_of_something = np.sum(sum_of_something)
    
    # formula 8/9
    dJ = np.array([])
    for i in range(0, len(training_data[:, 0])):
        temp = np.matmul(np.transpose(z),np.subtract(o, training_label))
        try:
            dJ = np.vstack((dJ, temp))
        except:
            dJ = temp
        
    obj_val = sum_of_something + (lambdaval/(2*len(training_data[:,0]))) * np.sum(np.sum(np.square(w1)) + np.sum(np.square(w2)))
    """
    
    """for i in range(0,n_input):
        inner_sum =0;
        y = training_label[i]
        for l in range(0,n_class):
            inner_sum += y * np.log(output_matrix[i][l]) + (1 - y) * np.log(1 - output_matrix[i][l])
                
        obj_val += inner_sum
    obj_val = obj_val*(-1/n_input)
    print("Obj Value: "+str(obj_val))
    #obj with regularization
    sum_hidden =0;
    sum_out = 0;
    for i in range(0,n_hidden):
        
        sum_hidden = np.sum(np.square(w1[i]))
    for j in range(0,n_class):
        sum_out = np.sum(np.square(w2[j]))
    obj_val_with_reg = (obj_val + lambdaval/(2*n_input))*(sum_hidden+sum_out)
    print("Obj Value With Regularization : "+ str(obj_val_with_reg ))
    print(output_matrix.shape)
    sum_ch = 0
    for l in range(0, n_class):
        sum_ch += np.sum(np.subtract(output_matrix[l,:], training_label[:])) + (np.multiply(w2[l], lambdaval))
        
    sum_ch /= n_input
    
    #gradient_h2o_matrix = np.array([])
    error = 0
    isFirst = True
    gradient_h2o_perInput = np.array([])
    for i in range(0,n_hidden):
        for j in range(0,n_class):
            if (isFirst):
                print(output_matrix.shape)
                print("---------")
                print(training_label.shape)
                print("-----------")
                print(h_matrix.shape)
                gradient_h2o_perInput = (output_matrix[i][j]-training_label[i])*h_matrix[i]
                isFirst = False
            else:
                gradient_h2o_perInput= np.vstack((gradient_h2o_perInput,(output_matrix[i][j]-training_label[i])*h_matrix[i]))
        error+=np.matmul(h_matrix[i],np.matmul(1-h_matrix[i],np.matmul(np.sum(np.matmul(gradient_h2o_perInput,w2[i])),training_data[i])))
    print("Gradient Error : " + str(error)) """
    
    
    """ for w in w2:
        w -= eta*error_output
        
    for w in w1:
        w -= eta*error_output
        
        
        
        
    # Your code here
    #
    #
    #
    #
    #
    
        
    
    for w in w2:
        w -= eta*error_output
        
    for w in w1:
        w -= eta*error_output
    """
    # w1, w2, obj_val, index = reduce(lambda label, label2: obj_helper(obj_helper((w1,w2,0,0),*args,label),*args,label2), training_label)
    
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


initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

opts = {'maxiter': 2}

minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
objval,objgrad = nnObjFunction(params, *args)

print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)
