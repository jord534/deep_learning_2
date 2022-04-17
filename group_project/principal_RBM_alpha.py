# Nom du script principal_RBM_alpha
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def lire_alpha_digit(indice_carac):
	mat = scipy.io.loadmat('binaryalphadigs.mat')
	X = []
	for carc in indice_carac:
		for k in range(39):
			X.append(mat['dat'][carc, :][k].reshape((1, -1))[0])
	return np.array(X)

def sigma(t):
	return 1/(1+np.exp(-t))

class RBM():
	def __init__(self, p, q):
		self.p = p
		self.q = q
		self.W = np.random.randn(p, q)*0.01
		self.a = np.zeros((p, 1))
		self.b = np.zeros((q, 1))

	def entre_sortie_RBM(self, V):
		n_batch = V.shape[0]
		b_augmente = np.repeat(self.b, n_batch, axis=1).T
		H = sigma(V@self.W+b_augmente)
		return H

	def sortie_entre_RBM(self, H):
		n_batch = H.shape[0]
		a_augmente = np.repeat(self.a, n_batch, axis=1).T
		V = sigma(H@self.W.T+a_augmente)
		return V


def train_RBM(RBM, X, nb_iter, batch_size, alpha):
	n = X.shape[0]
	p, q = RBM.p, RBM.q
	for i in range(nb_iter):
		shuffle(X)
		for iter_batch in range(0, n, batch_size):
			x_batch = X[iter_batch: min(n, iter_batch+batch_size), :]
			n_batch = x_batch.shape[0]
			v0 = x_batch

			p_v0 = RBM.entre_sortie_RBM(v0)
			h0 = (np.random.randn(n_batch, q) < p_v0)*1.
			p_h0 = RBM.sortie_entre_RBM(h0)
			v1 = (np.random.randn(n_batch, p) < p_h0)*1.
			p_v1 = RBM.entre_sortie_RBM(v1)	

			#Calcul des gradients
			grad_a = np.mean(v0-v1, axis=0).reshape(-1, 1)
			grad_b = np.mean(p_v0-p_v1, axis=0).reshape(-1, 1)
			grad_w = 0
			for k in range(n_batch):
				grad_w += (v0[k, :].reshape(-1, 1)@p_v0[k, :].reshape(1, -1)-
						v1[k, :].reshape(-1, 1)@p_v1[k, :].reshape(1, -1))							

			#Mise à jour des coefs
			RBM.a += alpha*grad_a
			RBM.b += alpha*grad_b
			RBM.W += alpha*grad_w/n_batch

		if i%10==0:
			#Afiiche la qualité de la reconstruction
			H = RBM.entre_sortie_RBM(X)
			X_reconstruit = RBM.sortie_entre_RBM(H) 
			print( 'Iter : {}   Erreur : {}'.format(i, 
				np.mean((X- X_reconstruit)**2)))
	return RBM

def generer_image_RBM(RBM, nb_iter_gibbs, nb_images):
	p, q = RBM.p, RBM.q
	iamges = []
	for k in range(nb_images):
		v = ((np.random.rand(p)<0.5)*1.).reshape((1,-1))
		for j in range(nb_iter_gibbs):
			h = (np.random.rand(q)<RBM.entre_sortie_RBM(v))*1.
			v = (np.random.rand(p)<RBM.sortie_entre_RBM(h))*1.
		iamges.append(v)
		plt.imshow(v.reshape(20, 16),cmap='gray', vmin=0, vmax=1)
		plt.show()
	return iamges