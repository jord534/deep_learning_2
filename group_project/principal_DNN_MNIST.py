
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import principal_RBM_alpha as rbm
import principal_DBN_alpha as dbn
import idx2numpy
from sklearn.preprocessing import OneHotEncoder


def calcul_softmax(RBM, X):
	val = RBM.entre_sortie_RBM(X)
	Z =  np.exp(val).sum(axis=1)
	proba = (np.exp(val).T/Z).T
	return proba

"""
def entree_sortie_reseau(DNN, X):
	res = np.array(X)
	for k in range(DNN.nb_couche):
		res_couche = DNN.reseau[k].entree_sortie_RBM(res[-1])
		res = np.append(res, np.array([res_couche]), axis=0)
	DNN.add_classification_layer(n_classes)
	prob = calcul_softmax(DNN.classification_layer, res[-1])
	res = np.append(res, np.array(prob), axis=0)
	return res
"""

def entree_sortie_reseau(DNN, input_data):
    layer_outputs=[np.array(input_data)] 
    layer_outputs.append(DNN.reseau[0].entre_sortie_RBM(input_data)) # input layer
    n_hidden_layers = DNN.nb_couche
    for layer in range(1,n_hidden_layers):
        layer_outputs.append(   DNN.reseau[layer].entre_sortie_RBM( 
                                layer_outputs[-1]) )
    layer_outputs.append(calcul_softmax(DNN.classification_layer, layer_outputs[-1] ))
    return layer_outputs

def cross_entropy(y_hat, y, eps=1e-8):
    n = y.shape[-1]
    return -np.sum(np.multiply(y,np.log(y_hat+eps)), axis = -1)

""""
def retropropagation(DNN, X, Y, nb_iter, batch_size, alpha):
	n = X.shape[0]
	for i in range(nb_iter):
		shuffle(X)
		for iter_batch in range(0, n, batch_size):
			x_batch = X[iter_batch: min(n, iter_batch+batch_size), :]
			n_batch = x_batch.shape[0]
			res_layers = np.array(entree_sortie_reseau(DNN, x_batch))
			gamma = (res_layers[-1,:] - Y)
			for l in range(1,DNN.nb_couche):
				gamma = (1-res_layers[-1-l])*res_layers[-1-l,:].W*(DNN.reseau[-l]@gamma.T)
				res_layers[-1-l].b -= alpha*gamma
				res_layers[-1-l].W -= alpha*res_layers[-1-l,:].T@gamma
		if i%10==0:
			p_hat = entree_sortie_reseau(DNN, X)[-1]
			loss = -np.sum(Y*np.log(p_hat))
			print( 'CrossEntropie : {}'.format(loss))
"""
def retropropagation(DNN, input_data, n_iterations, learning_rate, batch_size, data_size, data_labels):

	for i in range(n_iterations):
		shuffled_data = shuffle(input_data, data_labels) # same shuffling applied to images and labels
		shuffled_input, shuffled_labels = shuffled_data[0], shuffled_data[1]

		for iter_batch in range(0, data_size, batch_size):
			input_batch = shuffled_input[iter_batch: min(data_size, iter_batch+batch_size), :]
			labels_batch = shuffled_labels[iter_batch: min(data_size, iter_batch+batch_size), :]
			network_output = entree_sortie_reseau(DNN, input_batch) 
			classes_hat = network_output[-1] # softmaxed
			cost = cross_entropy(classes_hat, labels_batch) # loss/cost of the current batch
			
			### let's compute the gradients on this batch
			
			## first we compute d loss / da * da/dz for the ultimate layer 
			#loss_grad = np.asarray(-labels_batch/classes_hat) # gradient of Cross Entropy
			loss_grad = np.asarray(classes_hat-labels_batch)

			backpass = loss_grad
			## we'll start with the grads on class layer :
			grad_W = []
			grad_b = [] 
			a = network_output[-2]# a is the last sigmoid activated layer

			new_grad_W = np.einsum('ij,ik->ikj', a, backpass)
			new_grad_b = backpass

			grad_W.append(new_grad_W) # shape is (50,20,10)
			grad_b.append(new_grad_b) # shape is (50,10)

			backpass = np.dot(backpass,DNN.classification_layer.W.T) # shape is (50,20)
			## then we can iterate over the layers :
			for layer in range(1,DNN.nb_couche+1) : # this includes the input layer
				# sig grad:
				grad_sig = a*(1-a) # shape is (50,20)
				backpass = grad_sig*backpass # Hadamard product # shape is (50,20)

				# to have the grad on W_(layer), we need a_(layer-1)
				a = network_output[-layer-2]
				new_grad_W = np.einsum('ij,ik->ikj',a, backpass)
				new_grad_b = backpass

				grad_W.append(new_grad_W)
				grad_b.append(new_grad_b)

				backpass = np.dot(backpass, DNN.reseau[-layer].W.T) 

			### update the parameters
			#l_r = learning_rate
			#if i > 1 and abs(loss-loss_prev)<0.001 :
			#	l_r = 10*l_r
			#else :
			#	l_r = learning_rate
			DNN.classification_layer.W -= l_r * np.mean(grad_W[0], axis = 0).T
			DNN.classification_layer.b -= l_r * np.expand_dims(np.mean(grad_b[0], axis = 0), axis =-1)
			for layer in range(1,DNN.nb_couche+1) :
				DNN.reseau[-layer].W -= l_r * np.mean(grad_W[layer], axis =0).T
				DNN.reseau[-layer].b -= l_r * np.expand_dims(np.mean(grad_b[layer], axis =0), axis = -1 )
		
		# Compute Binary Cross Entropy on the whole set :
		loss = np.mean(cross_entropy(entree_sortie_reseau(DNN, shuffled_input)[-1], shuffled_labels), axis = 0)
		loss_prev = loss
		print("Epoch ", i+1, "/", n_iterations, " , loss : ", loss)

	return DNN	




def test_DNN(DNN_train, X_test, Y_test):
	p_hat = entree_sortie_reseau(DNN_train, X_test)[-1]
	y_pred = np.argmax(p_hat, axis=1)
	y_true = np.argmax(Y_test, axis=1)
	erreur = np.sum(y_pred != y_true)/len(y_true)
	return erreur
