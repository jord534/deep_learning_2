import numpy as np
import principal_RBM_alpha as bm
from scipy.special import softmax

def calcul_softmax(rbm, input_data):
    """
    will give the probabilities of input_data accoridng to rbm
    using softmax
    """ 
    output = rbm.W.dot(input_data) + rbm.a
    return softmax(output)

def entree_sortie_reseau(dnn, input_data):
    layer_outputs=[] 
    layer_outputs.append(bm.entree_sortie_RBM(dnn.input, input_data))
    n_hidden_layers = len(dnn.hidden_layers)
    for layer in range(n_hidden_layers):
        layer_outputs.append(bm.entree_sortie_RBM(   dnn.hidden_layers[layer], 
                                                            layer_outputs[-1]) )
    layer_outputs.append(calcul_softmax(dnn.classification_layer, layer_outputs[-1] ))
    return layer_outputs


def binary_cross_entropy(y_hat, y):
    n = y.shape[-1]
    return 1/n * np.sum(y*np.log(y_hat), axis = -1)

def retropropagation(dnn, n_iterations, learning_rate, batch_size, data_size, data_labels):
    
    # let's code the algoirthm for one layer to the next
    network_output = entree_sortie_reseau(dnn, input_data) # !!! same problem : which data to train on? retropropagation needs to know
    classes_hat = network_output[-1]
    cost = binary_cross_entropy(classes_hat, data_labels) # !!! : need to one hot encode labels

    return 0