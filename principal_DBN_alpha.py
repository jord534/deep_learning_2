import numpy as np
import principal_RBM_alpha as bm
import matplotlib.pyplot as plt

class DNN:
    def __init__(self, input_rbm, n_neurons, n_hidden_layers, output_classification) :
        self.layer_size = n_neurons
        self.input = input_rbm
        self.hidden_layers = [] # a list of RBMs
        for i in range(n_hidden_layers):
            self.hideen_layers.append(bm.init_RBM(self.layer_size,self.layer_size))
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

def pretrain_DNN(dnn, n_iterations, learning_rate, batch_size, data_size):

    dnn.input = bm.train_RBM(dnn.input, n_iterations, learning_rate, batch_size, data_size)
    for layer in range(len(dnn.hidden_layers)):
        # !!!! need to fix the problem : train_RBM doesn't know which data to train on
        # !!!! when this is fixed, give each layer the previous layer's output to start training on
        # !!!! this needs to be fixed in principal_RBM_alpha
        dnn.hidden_layers[layer] = bm.train_RBM(dnn.hidden_layers[layer], n_iterations, learning_rate, batch_size, dnn.layer_size)
    # we don't train the classification layer just yet
    return dnn

def generer_image_DBM(dnn, n_iterations, n_images):
    images = []
    input_size = dnn.a.shape
    output_size = dnn.b.shape
    for i in range(n_img):
        v = (np.random.rand(input_size)<0.5).reshape(1,-1)
        for k in range(n_it) : # Gibbs sampling
            h = (np.random.rand(output_size)<entree_sortie_(dnn, v*.))*1.
            v = (np.random.rand(input_size)<sortie_entree_dnn(dnn, h*1.))*1.
        images.append(v)
        plt.imshow(v.reshape(20,16), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()
    return images
    return 0