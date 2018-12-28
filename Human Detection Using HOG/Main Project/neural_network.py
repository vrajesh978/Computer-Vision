import numpy as np

#sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# derivative of sigmoid function
def d_Sigmoid(x):
    return x * (1 - x)

#ReLu function
def ReLu(x):
    return x * (x > 0)
#Derivative of ReLu function
def d_ReLu(x):
    return 1. * (x > 0)


def trainNeuralNetwork(X,Y,no_hidden_neurons):
    '''
    parameter : 
        @param1: X, input list of all training sample's feature vector. 
        @param2: Y, list of actual training label. 
        @param3: no_hidden_neurons, number of hidden layer neurons.
        @return : dictionary , it contains model's parmaeters (weight and bias)
    '''
    np.random.seed(1)
    '''random initialization of weight and bias'''
    w1 = np.random.randn(no_hidden_neurons, len(X[0])) * 0.01
    b1 = np.zeros((no_hidden_neurons,1))
    w2 = np.random.randn(1,no_hidden_neurons) * 0.01
    b2 = np.zeros((1,1))
    
    dictionary = {} #This will contain updated weight and bias.    
    cost_avg = 0.0 # average cost of our network
    old_cost = 0.0

    """ Our neural network will train maximum up to 200 epoch. 
    if cost between two epochs is less than 0.02, we will stop. 
    Because we know that our weights does not change too much."""
    for i in range(0,200):
        cost = 0.0
        for j in range(0,len(X)):
            q = X[j]    #getting feature vector from the list.
            '''Neural network train'''
            #forward pass
            z1 = w1.dot(q)+ b1   
            a1 = ReLu(z1)
            z2 = w2.dot(a1) + b2
            a2 = sigmoid(z2)
            cost += (1.0/2.0)*(np.square((a2-Y[j])))  #findng the cost of the every image and sum their cost.

            # Backward Propogation
            dz2 = (a2-Y[j])  *  d_Sigmoid(a2)
            dw2 = np.dot(dz2,a1.T)
            db2 = np.sum(dz2,axis=1, keepdims=True)

            dz1 = w2.T.dot(dz2) * d_ReLu(a1)
            dw1 =  np.dot(dz1,q.T)
            db1 =  np.sum(dz1,axis=1, keepdims=True)

            #updating weights. Here 0.01 is the learning rate
            w1 = w1 - 0.01*dw1  
            w2 = w2 - 0.01*dw2
            b1 = b1 - 0.01*db1
            b2 = b2 - 0.01*db2
            
        cost_avg = cost/len(X)    #taking average cost
        print("Epoch = ",i+1,"cost_avg = ",cost_avg[0][0])
        dictionary = {'w1':w1,'b1':b1,'w2':w2,'b2':b2} #save our updated weights. So that we can use them while testing.
        # if cost between two epochs is less than 0.0001, we will stop. Because we know that our weights does not change too much.
        if(abs(old_cost-cost_avg)<=0.0001):   
            return dictionary
        else:
            old_cost = cost_avg
    return dictionary

def saveModelFile(dictionary,name):
    """
    Saving our model's parameter so we dont have to start all the process again.
    @param1: dictionary, containg our trained model parameters
    """
    np.save(str(name)+".npy",dictionary)
    print("Successfully saved model file as",str(name),".npy")

def loadModelFile(name):
    """
        Loading our modelfile.
        @return : dictionary, containing our trained model parameters
    """
    print("Loading model file")
    dictionary = np.load(str(name)+".npy")
    print("Successfully loaded model files")
    return dictionary[()]

def predict(X_test,dictionary):
    """
        Predict the newly seen data.
        @param1: X_test, new data that is not seen by our model yet.
        @param2: dictionary, containing our trained model parameters
        @return : neural network prediction.
    """
    w1,w2,b1,b2 = dictionary['w1'],dictionary['w2'],dictionary['b1'],dictionary['b2'] # getting the data from the dictionary.
    z1 = w1.dot(X_test)+ b1   
    a1 = ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    return a2

def accuracy(neural_network_output,y_test):
    """
        Find the accuracy of the model.
        @param1: neural_network_output. neural network prediction.
        @param2: y_test. 
        @return :  accuracy in terms of percentage.
    """
    count = 0
    for no,ao in zip(neural_network_output,y_test):
        '''if neural network's output is > 0.5, 
        it means our neural network has detected that 
        there is a human in the image other wise there is not human in the image.'''
        if no[0] > 0.5:    
            count+=abs(1.0-ao[0])
        else:
            count+=abs(0.0-ao[0])
    return (((len(y_test)-count)/len(y_test))*100)[0]

