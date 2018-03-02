import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from functools import reduce

eta = 0.05

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


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    #Plugin the sigmoid function from lecture
    #1/(1+e^(-net) where net is input z
    #np.exp is python e operator
    return  1/(1 + np.exp(-1 * z))


def preprocess():
    #Someone on piazza said to do this to fix error
    n_valid = 5000
    #end piazza tip
    
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Things to do for preprocessing step:
     - remove features that have the same value for all data points
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - divide the original data set to training, validation and testing set"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    train_data = np.concatenate((mat['train0'], mat['train1'],
                                 mat['train2'], mat['train3'],
                                 mat['train4'], mat['train5'],
                                 mat['train6'], mat['train7'],
                                 mat['train8'], mat['train9']), 0)
    train_label = np.concatenate((np.ones((mat['train0'].shape[0], 1), dtype='uint8'),
                                  2 * np.ones((mat['train1'].shape[0], 1), dtype='uint8'),
                                  3 * np.ones((mat['train2'].shape[0], 1), dtype='uint8'),
                                  4 * np.ones((mat['train3'].shape[0], 1), dtype='uint8'),
                                  5 * np.ones((mat['train4'].shape[0], 1), dtype='uint8'),
                                  6 * np.ones((mat['train5'].shape[0], 1), dtype='uint8'),
                                  7 * np.ones((mat['train6'].shape[0], 1), dtype='uint8'),
                                  8 * np.ones((mat['train7'].shape[0], 1), dtype='uint8'),
                                  9 * np.ones((mat['train8'].shape[0], 1), dtype='uint8'),
                                  10 * np.ones((mat['train9'].shape[0], 1), dtype='uint8')), 0)
    test_label = np.concatenate((np.ones((mat['test0'].shape[0], 1), dtype='uint8'),
                                 2 * np.ones((mat['test1'].shape[0], 1), dtype='uint8'),
                                 3 * np.ones((mat['test2'].shape[0], 1), dtype='uint8'),
                                 4 * np.ones((mat['test3'].shape[0], 1), dtype='uint8'),
                                 5 * np.ones((mat['test4'].shape[0], 1), dtype='uint8'),
                                 6 * np.ones((mat['test5'].shape[0], 1), dtype='uint8'),
                                 7 * np.ones((mat['test6'].shape[0], 1), dtype='uint8'),
                                 8 * np.ones((mat['test7'].shape[0], 1), dtype='uint8'),
                                 9 * np.ones((mat['test8'].shape[0], 1), dtype='uint8'),
                                 10 * np.ones((mat['test9'].shape[0], 1), dtype='uint8')), 0)
    test_data = np.concatenate((mat['test0'], mat['test1'],
                                mat['test2'], mat['test3'],
                                mat['test4'], mat['test5'],
                                mat['test6'], mat['test7'],
                                mat['test8'], mat['test9']), 0)
   # remove features that have same value for all points in the training data
    # convert data to double
    # normalize data to [0,1]

    # Split train_data and train_label into train_data, validation_data and train_label, validation_label
    # replace the next two lines
    
    #Remove features
    temp= train_data[:, ~np.all(train_data == train_data[0,:], axis=0)]
    #Normalize
    temp = temp/255.0
    #Create Random Row indecs this is to split data randomly
    ind= np.random.permutation(temp.shape[0])
    #Shuffle based on ind
    temp = np.take(temp,ind,axis = 0)
    #Shuffle labels to match this
    #temp_labels = np.take(train)
   
    # Split train_data and train_label into train_data, validation_data and train_label, validation_label
    
    #Giving 15% to validation_data
    #85% to training data
    # Percentage is of train_data
    validation_data = temp[:int(temp.shape[0]*.15),:]
    train_data = temp[int(temp.shape[0]*.15):,:]
    
    temp_labels = np.take(train_label,ind,axis =0)
    #Splitting the labels
    validation_label = temp_labels[:int(temp_labels.shape[0]*.15),:]
    train_label = temp_labels[int(temp_labels.shape[0]*.15):,:]
    
    

    print("preprocess done!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label


'''def obj_helper(*inputs, *args,y):
    w1,w2,obj_val,j = inputs
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    
    labels = np.array([])
    # Your code here
    bias_td = np.append(j, [1])
    h = np.array([])
    for i in w1:
        a = np.matmul(np.transpose(i), bias_td)
        z = sigmoid(a)
        h = np.append(h, z)
        
    h = np.append(h, 1)
    
    for i in w2:
        a = np.matmul(np.transpose(i), h)
        z = sigmoid(a)
        labels = np.append(labels, z)
    
    outputs=labels
    error_output = 0
    for l in outputs:
        error_output += y * np.log(l) + (1 - y) * np.log(1 - l)
        
    error_output *= -1
    obj_val+=error_output
    for w in w2:
        w -= eta*error_output
        
    for w in w1:
        w -= eta*error_output
    return w1,w2,obj_val,j+1
'''    


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
    
    
    
    
    
    #use this line for objective func
    output_matrix = np.array([])
    h_matrix = np.array([])
    isFirst= True
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
            
        h_matrix = np.append(h_matrix,h,axis =0)
        if(isFirst):
            output_matrix = output_per_input
            isFirst = False;
        else:
            output_matrix = np.vstack((output_matrix,output_per_input))
       
    
    obj_val = 0
    for i in range(0,n_input):
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
    print("Gradient Error : " + str(error)) 
    for w in w2:
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
        
    w1, w2, obj_val, index = reduce(lambda label, label2: obj_helper(obj_helper((w1,w2,0,0),*args,label),*args,label2), training_label)
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((w1.flatten(), w2.flatten()),0)
    

    return (obj_val, obj_grad)



def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    labels = np.array([])
    
    #use this line for objective func
    outputs_matrix = np.array([])
    bias_td = np.append(j, [1])
    h = np.array([])
    for i in w1:
        a = np.matmul(np.transpose(i), bias_td)
        z = sigmoid(a)
        h = np.append(h, z)
        
    h = np.append(h, 1)
    
    for i in w2:
        a = np.matmul(np.transpose(i), h)
        z = sigmoid(a)
        labels = np.append(labels, z)
    
    outputs=labels
    error_output = 0
    for l in outputs:
        error_output += y * np.log(l) + (1 - y) * np.log(1 - l)
        
    error_output *= -1
    obj_val+=error_output
    for w in w2:
        w -= eta*error_output
        
    for w in w1:
        w -= eta*error_output

    

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.01

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))




n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnObjFunction(params, *args)
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)


#Run nnpredict once
#Run nnobject pass new weights to nnPredict do this at the bottom of nnobject






# Test the computed parameters

#print(train_data.shape)
#predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

#print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

#predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

#print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

#print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
