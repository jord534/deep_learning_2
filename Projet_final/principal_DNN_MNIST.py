import numpy as np
from sklearn.utils import shuffle
from principal_RBM_alpha import *
from principal_DBN_alpha import *


def softmax(X):
	"""Retourne le softmax d'un tableau de dimension 2.
	"""
	Z = np.exp(X)
	div = np.sum(Z, axis=1)
	return Z / div.reshape(-1, 1)

def compute_softmax(X, rbm):
	"""Retourne les probabilités calculées sur les unités
	de sortie du RBM à partir de la fonction softmax.
	"""
	Z = (X @ rbm.W) + rbm.b
	return softmax(Z)


def entree_sortie_reseau(dnn, X):
	"""Retourne les sorties sur chaque couche du dnn (couche d’entrée inclue)
	ainsi que les probabilités sur les unités de sortie.
	"""
	res = []
	H = X
	# On ajoute la sortie de chaque couche au résultat
	for i in range(len(dnn.weights)-1):
		p, q = dnn.weights[i].shape
		rbm = RBM(p, q)
		rbm.W = dnn.weights[i]
		rbm.a = dnn.biases[i]
		rbm.b = dnn.biases[i+1]
		H = entree_sortie_RBM(rbm, H)
		res.append(H)
	# Enfin on ajoute les probas sur les unités de sortie
	i += 1
	p, q = dnn.weights[i].shape
	rbm = RBM(p, q)
	rbm.W = dnn.weights[i]
	rbm.a = dnn.biases[i]
	rbm.b = dnn.biases[i+1]
	probas = compute_softmax(res[-1], rbm)
	res.append(probas)
	return res



def cross_entropy(dnn, X, y, e=1e-8):
	"""Retourne la cross-entropy entre les labels réels
	et les probas en sortie du DNN.
	"""
	probas = entree_sortie_reseau(dnn, X)[-1]
	cross_ent = -y * np.log(probas + e)
	return np.sum(cross_ent) / y.shape[0]

 
def retropropagation(dnn, X, y, batch_size, n_iter, learning_rate):
	"""Retourne un DNN entraîné par rétropropagation.
	"""
	X_ = X.copy()
	y_ = y.copy()
	n = X_.shape[0]
	
	losses = []
	losses.append(cross_entropy(dnn, X_, y_))

	len_weights = len(dnn.weights)
	
	for i in range(n_iter):
		X_, y_ = shuffle(X_, y_)
		
		for j in range(1, n, batch_size):
			H = entree_sortie_reseau(dnn, X_[j: min(j+batch_size, n), :])
			
			for layer in range(len_weights-1, -1, -1):
				# Première couche du réseau
				if (layer == 0):
					input_ = X_[j: min(j+batch_size, n), :]
				else:
					input_ = H[layer-1]

				input_ = np.c_[np.ones(input_.shape[0]), input_]

				# Dernière couche du réseau
				if (layer == len_weights-1):
					dz = H[-1] - y_[j: min(j+batch_size, n), :]
				else:
					da = H[layer] * (1 - H[layer])
					dz = da * (dz @ dnn.weights[layer+1].T)

				grads = (input_.T @ dz) / input_.shape[0]
				grad_b = grads[0, :]
				grad_W = grads[1:, :]

				dnn.weights[layer] -= learning_rate * grad_W
				dnn.biases[layer+1] -= learning_rate * grad_b
        
		losses.append(cross_entropy(dnn, X_, y_))

		if (i % int(n_iter/10) == 0):
			print('Iteration Rétropropagation (%d/%d) \t cross_entropy : %.4f' %(i, n_iter, losses[-1]))
    
	return dnn, losses


def predict(dnn, X):
	"""Retourne les predictions du DNN sur les données.
	"""
	return np.argmax(entree_sortie_reseau(dnn, X)[-1], axis=1)


def test_DNN(dnn, X, y):
	"""Retourne le taux d'erreur du DNN sur les données.
	"""
	y_pred = predict(dnn, X)
	accuracy = np.mean(1 * (np.argmax(y, axis=1) == y_pred))
	return accuracy
