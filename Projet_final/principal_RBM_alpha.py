# Nom du script principal_RBM_learning_rate
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def lire_alpha_digit(file_name: str, indices: list):
	"""Récupère les données des alphaDigits sous forme matricielle
	pour les indices des caractères correspondants.
	"""
	X_alphadigs = scipy.io.loadmat(file_name)['dat']
	X = X_alphadigs[indices, :].flatten()
	X = np.array([x.flatten() for x in X])
	return X


class RBM():
	def __init__(self, p, q):
		"""Construit et initialise les poids et les biais d’un RBM
		"""
		self.W = 0.01 * np.random.normal(0, 0.1, size=(p, q))
		self.a = np.zeros((1, p))                             
		self.b = np.zeros((1, q))


def sigmoid(t):
	"""Retourne la valeur de la fonction sigmoïde pour un élément donné.
	"""
	return 1 / (1 + np.exp(-t))

def entree_sortie_RBM(rbm, V):
	"""Retourne la valeur des unités de sortie calculées
	à partir de la fonction sigmoïde.
	"""
	return sigmoid((V @ rbm.W) + rbm.b)
 
def sortie_entree_RBM(rbm, H):
	"""Retourne la valeur des unités d’entrée
	à partir de la fonction sigmoïde.
	"""
	return sigmoid((H @ rbm.W.T) + rbm.a)


def train_RBM(rbm, X, nb_iter, batch_size, learning_rate):
	"""Retourne un RBM entraîné de manière non supervisée
	par l’algorithme Contrastive-Divergence-1
	"""
	n = X.shape[0]
	p, q = rbm.W.shape
	losses = np.zeros(nb_iter)

	for i in range(nb_iter):
		shuffle(X)

		for iter_batch in range(0, n, batch_size):
			x_batch = X[iter_batch: min(n, iter_batch+batch_size), :]
			n_batch = x_batch.shape[0]

			v0 = x_batch
			p_v0 = entree_sortie_RBM(rbm, v0)
			h0 = (np.random.rand(n_batch, q) < p_v0) * 1.
			p_h0 = sortie_entree_RBM(rbm, h0)
			v1 = (np.random.rand(n_batch, p) < p_h0) * 1.
			p_v1 = entree_sortie_RBM(rbm, v1)

			#Calcul des gradients
			grad_a = np.mean(v0 - v1, axis=0)
			grad_b = np.mean(p_v0 - p_v1, axis=0)
			grad_W = (v0.T @ p_v0) - (v1.T @ p_v1)						

			#Mise à jour des coefs
			rbm.a += learning_rate * grad_a
			rbm.b += learning_rate * grad_b
			rbm.W += learning_rate * grad_W / n_batch

		#MSE
		H = entree_sortie_RBM(rbm, X)
		X_reconstruit = sortie_entree_RBM(rbm, H) 
		mse = np.mean((X- X_reconstruit)**2)
		losses[i] = mse

		if i%100 == 0:
			#Afiiche la qualité de la reconstruction
			print( 'Iteration RBM (%d/%d) \t  MSE : %.4f' %(i, nb_iter, mse))

	return rbm, losses


def generer_image_RBM(rbm, nb_iter_gibbs, nb_images, indices):
	"""Génère des échantillons suivant un RBM.
	"""
	p, q = rbm.W.shape
	images = []

	fig, axs = plt.subplots(1, nb_images, figsize=(15,3))
	fig.suptitle('Echantillons générés par un RBM pour les indices %s' %indices)

	for k in range(nb_images):
		v = ((np.random.rand(p) < 0.5) * 1.).reshape((1, -1))

		for j in range(nb_iter_gibbs):
			h = (np.random.rand(q) < entree_sortie_RBM(rbm, v)) * 1.
			v = (np.random.rand(p) < sortie_entree_RBM(rbm, h)) * 1.

		images.append(v)
		axs[k].imshow(v.reshape(20, 16), cmap='gray', vmin=0, vmax=1)

	plt.show()
	return images