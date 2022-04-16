import numpy as np
import principal_RBM_alpha as bm

class DNN:
    def __init__(self, input_rbm, n_neurons, n_hidden_layers, output_classification) :
        self.input = input_rbm
        self.hidden_layers = [] # a list of RBMs
        for i in range(n_hidden_layers):
            self.hideen_layers.append(bm.init_RBM(n_neurons,n_neurons))
        self.classification_layer = output_classification




def init_DNN(network_size):
    # network_size is a tuple with rule :
    # (number_of_neurons,number_of_layers)
    n_neurons = network_size[0]
    n_layers = network_size[1]
    dnn = DNN(
            input_rbm= bm.init_RBM(None,n_neurons), 
            n_neurons= n_neurons,
            n_hidden_layers= n_layers-2, 
            output_classification= bm.init_RBM(n_neurons, None) 
            )
    return dnn

def pretrain_DNN():
    
    return 0