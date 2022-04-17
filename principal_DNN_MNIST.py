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
    return -np.sum(y*np.log(y_hat), axis = -1)

def retropropagation(dnn, n_iterations, learning_rate, batch_size, data_size, data_labels):
    
    network_output = entree_sortie_reseau(dnn, input_data) # !!! same problem : which data to train on? retropropagation needs to know
    classes_hat = network_output[-1]
    cost = binary_cross_entropy(classes_hat, data_labels)

    n_classes = len(data_labels)
    C_grad_w = []
    C_grad_b = []
    sftmx_grad_z = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        sftmx_grad_z[i,i] = classes_hat[i]*(1-classes_hat[i]) 
        for j in range(1, n_classes):
            sftmx_grad_z[i,j] = -classes_hat[i]*classes_hat[j]
            sftmx_grad_z[j,i] = sftmx_grad_z[i,j]


    curr_prod = (-data_labels/classes_hat).dot(sftmx_grad_z) 

    #for the classification layer :
    C_grad_w.append( curr_prod.dot( network_output[-2] ) )

    # for hidden layers
    for k in range(2,len(network_output)):
        curr_w = 
        curr_prod = curr_prod.dot(curr_w)
        curr_prod = curr_prod.dot(  network_output[-k]*(1-network_output[-k])  )
        new_grad_w = 
        C_grad_w.append(new_grad_w)
        curr_prod = 

        curr_b = 

    
    return 0