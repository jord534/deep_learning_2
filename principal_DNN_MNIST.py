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

def retropropagation(dnn, n_iterations, ):

    return 0