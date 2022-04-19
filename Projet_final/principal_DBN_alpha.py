import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from principal_RBM_alpha import *

class DNN():
	def __init__(self, input_dim: int, output_dim: int, hidden_dims: list):
		"""Construit et initialise les poids et les biais d’un DNN.
		"""
		self.weights = {}
		self.biases = {}
		layer_sizes = [input_dim] + hidden_dims + [output_dim]

		for i in range(len(layer_sizes)-1):
			W, a, b = self.init_layer(layer_sizes[i], layer_sizes[i+1])
			self.weights[i] = W
			self.biases[i] = a
			self.biases[i+1] = b

	def init_layer(self, input_size, output_size):
		W = np.random.normal(0, 0.1, (input_size, output_size))
		a = np.zeros((1, input_size))
		b = np.zeros((1, output_size))
		return W, a, b


def pretrain_DNN(dnn, X, nb_iter, batch_size, learning_rate):
	"""Entraîne de manière non supervisée un DBN (Greedy layer wise procedure).
	"""
	H = X
	shuffle(H)
	# On entraîne chaque couche du DNN
	for i in range(len(dnn.weights)):
		print("Pré-entrainement de la couche (%d/%d) du DNN" %(i+1, len(dnn.weights)))
		p, q = dnn.weights[i].shape
		rbm = RBM(p, q)
		rbm.W = dnn.weights[i]
		rbm.a = dnn.biases[i]
		# Entraînement d'un RBM
		rbm, _ = train_RBM(rbm, H, nb_iter, batch_size, learning_rate)
		# On met à jour les coefficients de la couche du DNN
		dnn.weights[i] = rbm.W
		dnn.biases[i] = rbm.a
		dnn.biases[i+1] = rbm.b
		H = entree_sortie_RBM(rbm, H)

	return dnn
 
def entree_sortie_DBN(dnn, V):
	"""Retourne la valeur des unités de sortie du DNN.
	"""
	H = V
	for i in range(len(dnn.weights)):
		p, q = dnn.weights[i].shape
		rbm = RBM(p, q)
		rbm.W = dnn.weights[i]
		rbm.a = dnn.biases[i]
		rbm.b = dnn.biases[i+1]
		H = entree_sortie_RBM(rbm, H)
	return H


def sortie_entree_DBN(dnn, H):
	"""Retourne la valeur des unités de sortie du DNN.
	"""
	V = H
	for i in range(len(dnn.weights)-1, -1, -1):
		p, q = dnn.weights[i].shape
		rbm = RBM(p, q)
		rbm.W = dnn.weights[i]
		rbm.a = dnn.biases[i]
		rbm.b = dnn.biases[i+1]
		V = sortie_entree_RBM(rbm, V)
	return V


def generer_image_DBN(dnn, n_imgs, n_iter, indices):
	"""Génère des échantillons suivant un DBM.
	"""
	p, q = dnn.weights[0].shape[0], dnn.weights[len(dnn.weights)-1].shape[1]
	images = []

	fig, axs = plt.subplots(1, n_imgs, figsize=(15,3))
	fig.suptitle('Echantillons générés par un DBN pour les indices %s' %indices)

	v = (np.random.rand(n_imgs, p) < 0.5) * 1

	for j in range(n_iter):
		h = (np.random.rand(n_imgs, q) < entree_sortie_DBN(dnn, v))
		v = (np.random.rand(n_imgs, p) < sortie_entree_DBN(dnn, h))
	
	for i in range(n_imgs):
		images.append(v[i])
		axs[i].imshow(v[i].reshape(20, 16), cmap='gray')
	
	plt.show()
	return images