import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import principal_RBM_alpha as rbm
import principal_DBN_alpha as dbn
import idx2numpy
from sklearn.preprocessing import OneHotEncoder

##############################
######### AlphaDigit #########
##############################
p, q = 320, 50
nb_iter = 100
alpha = 0.1
batch_size = 50
n = 72
nb_iter_gibbs = 1000
nb_couche = 2


#Cahrge les données
X = rbm.lire_alpha_digit([2, 7]) #(72, 320)

"""
#RBM
#initialisation
RBM = rbm.RBM(p, q)
#train
rbm.train_RBM(RBM, X, nb_iter, batch_size, alpha)
#Test
rbm.generer_image_RBM(RBM, nb_iter_gibbs, 1)
"""


#DBM
#DBN = dbn.DNN(nb_couche, p, q)
#dbn.pretrain_DNN(DBN, X, nb_iter, batch_size, alpha)
#dbn.generer_image_DBN(DBN, nb_iter_gibbs, 1)





##############################
############ MNIST ###########
##############################


file = 'train-images.idx3-ubyte'
X = (idx2numpy.convert_from_file(file).reshape((60000,-1,1))[:,:,0]>127)*1.
X = X[:2000, :]
file = 'train-labels.idx1-ubyte'
Y = idx2numpy.convert_from_file(file)
Y = Y[:2000].reshape((-1, 1))
Y = OneHotEncoder(sparse = False).fit_transform(Y)



image_size = 28
num_images = 5
X_train = X[:1600,:]
Y_train = Y[:1600,:]
X_test = X[1601:,:]
Y_test = Y[1601:,:]

# DNN pretraining unsupervised :
n_layers = 2
alpha  = 0.5
input_dim = X_train.shape[-1]
n_neurons = 150
batch_size = 50
DNN = dbn.DNN(n_layers, input_dim, n_neurons)
#print(" Pre entrainement des ", n_layers, " couches du DNN")
#DNN = dbn.pretrain_DNN(DNN, X, nb_iter, batch_size, alpha)

# n_classes needs to be Y.shape[-1]
n_classes = Y.shape[-1]

#DNN is a DBN with a classification layer trained for such purpose :
DNN.add_classification_layer(n_classes)

import principal_DNN_MNIST as dnn
learning_rate = 0.4
nb_iter = 150
DNN = dnn.retropropagation(DNN, X_train, nb_iter, learning_rate, batch_size, X_train.shape[0], Y_train )

print( "Erreur de test :", 100*dnn.test_DNN(DNN, X_test, Y_test), "%")