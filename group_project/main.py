import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import principal_RBM_alpha as rbm
import principal_DBN_alpha as dbn
import idx2numpy


##############################
######### AlphaDigit #########
##############################
p, q = 320, 50
nb_iter = 100
alpha = 0.1
batch_size = 10
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
DBN = dbn.DNN(nb_couche, p, q)
dbn.pretrain_DNN(DBN, X, nb_iter, batch_size, alpha)
dbn.generer_image_DBN(DBN, nb_iter_gibbs, 1)





##############################
############ MNIST ###########
##############################

"""
file = 'train-images.idx3-ubyte'
X = (idx2numpy.convert_from_file(file).reshape((60000,-1,1))[:,:,0]>127)*1.
X = X[:1000, :]
file = 'train-labels.idx1-ubyte'
Y = idx2numpy.convert_from_file(file)
Y = Y[:1000].reshape((-1, 1))
Y = OneHotEncoder().fit_transform(Y)

image_size = 28
num_images = 5
"""