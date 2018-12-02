from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np


def ReLu(x):
    return x * (x > 0)

def ReLuDerivative(x):
    return 1. * (x > 0)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

def compute_multiclass_loss(Y, Y_hat,type):
    """Function to compute loss between two epoches 
        @param1 : Y, given training class
        @param2 : Y_hat, predicated class by our network
        @param3 : type , if type = 2 , its  Cross-entropy loss, or log loss
                        if type = 1 , its square error loss function. 
        @return : L, which has cost at epoch i
    """

    m = Y.shape[1]
    if type==2:
        L_sum = (-1)*np.sum(np.multiply(Y, np.log(Y_hat)))
    elif type == 1:
        L_sum = np.sum((Y_hat-Y)**2)
    L = (1/m) * L_sum
    return L

def build_nn(X,Y,n_classes,no_hidden_neurons,num_passes = 1000, print_loss=False):
    
    """
        @param1 : X, input of training example. Dimension [no_of_features x no_of_training_example]
        @param2 : Y, class . Dimension [no_of_classes x no_of_training_example]
        @param3 : n_classes, no of classes
        @param4 : no_hidden_neurons, no of neuros in hidden layer.
        @param5 : num_passes , no of epochs. Ny defualt it is set to 1000.
        @param6 : print_loss , if True , it will print loss at each 100th epochs. By default it is false.
        @return : model, contains weights which is set by our model.
    """
    # initiliazed our model parameters to random values.
    np.random.seed(0)
    w1 = np.random.randn(no_hidden_neurons,X.shape[0])  # wieght for hidden layer
    b1 = np.zeros((no_hidden_neurons,1)) # bias for hidden layer
    w2 = np.random.randn(n_classes,no_hidden_neurons)  # weight for output layer
    b2 = np.zeros((n_classes,1)) # bias for output layer
    m = Y.shape[1]  #getting training example.
    model = {}      #save weights parameters to model directory
    for i in range(num_passes):
        #forward pass of neural network.
        z1 = np.matmul(w1,X)+b1   # calculate z1 = w1.X + b1
        a1 = ReLu(z1)  # relu function    
        z2 = np.matmul(w2,a1)+b2  # calculate z2 = w2.a1 + b2
        probs = np.exp(z2) / np.sum(np.exp(z2), axis=0) #softmax function
        
        new_cost = compute_multiclass_loss(Y,probs,1) # calculate loss of our training at every epoch.
        
        '''
            if loss between two epochs is too small, we know that we have trained our network successfully. 
            So there is no point to calculate new weights because it does not change much. 
            It means that we have fitted our network. 
            That's why, We return our model parameter even if all the epoch haven't finished.
        '''
        if i > 0:
            if abs(cost-new_cost) <= 0.0000000001:
                return model

        cost = new_cost

        #backword pass
        dz2 = probs-Y 
        dw2 = (1./m) * np.matmul(dz2,a1.T)
        db2 = (1./m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(w2.T,dz2) * ReLuDerivative(z1)
        dw1 = (1./m) * np.matmul(dz1,X.T) 
        db1 = (1./m) * np.sum(dz1,axis=1,keepdims=True)
        
        
        #updating weights
        w1 = w1 - dw1
        w2 = w2 - dw2
        b1 = b1 - db1
        b2 = b2 - db2
        
        if (print_loss and i % 100 == 0):
            print("Epoch", i, "cost: ", cost)

        model = {'w1':w1,'b1':b1,'w2':w2,'b2':b2} #save model weights at every epoch.
    print("final cost = ",cost)
    return model

def predict(X,model):
    """ This predict function predict class that the input is belong to.
        @param1 : X , input vector dimension [1*]
        @param2 : model, trained model which is dictonary containing weights.
        @return : class of the input where it belong.
    """
    #getting model weights that we have trained.
    w1,w2,b1,b2 = model['w1'], model['w2'],model['b1'],model['b2']
    z1 = np.matmul(w1,X) + b1
    a1 = ReLu(z1)
    z2 = np.matmul(w2,a1) + b2 
    probs = np.exp(z2) / np.sum(np.exp(z2), axis=0,keepdims=True)
    return np.argmax(probs, axis=0)
