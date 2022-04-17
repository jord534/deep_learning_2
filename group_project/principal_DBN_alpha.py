import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import principal_RBM_alpha as rbm


class DNN():
	"""docstring for DNN"""
	def __init__(self, nb_couche, p, q):
		self.p = p
		self.q = q
		self.nb_couche = nb_couche
		self.reseau =[rbm.RBM(q, q) for k in range(nb_couche-1)]
		self.reseau.insert(0,rbm.RBM(p, q))
	
	# added : 
	def add_classification_layer(self, n_classes):
		self.n_classes = n_classes
		self.classification_layer = rbm.RBM(self.q ,n_classes)

def pretrain_DNN(DNN, X, nb_iter, batch_size, alpha):
	data = X
	for couche in range(DNN.nb_couche):
		DNN.reseau[couche] = rbm.train_RBM(DNN.reseau[couche], data, nb_iter, batch_size, alpha)
		data = DNN.reseau[couche].entre_sortie_RBM(data)
	return DNN

def generer_image_DBN(DNN, nb_iter_gibbs, nb_images):
	p, q = DNN.p, DNN.q
	images = []
	for k in range(nb_images):
		v = ((np.random.rand(p)<0.5)*1.).reshape((1, -1))
		for j in range(nb_iter_gibbs):
			h = (np.random.rand(q)<DNN.reseau[0].entre_sortie_RBM(v))*1.
			for couche in range(1, DNN.nb_couche):
				h = (np.random.rand(q)<DNN.reseau[couche].entre_sortie_RBM(h))*1.
			v = (np.random.rand(q)<DNN.reseau[DNN.nb_couche-1].sortie_entre_RBM(h))*1.
			for couche in range(DNN.nb_couche-1, 0, -1):
				v = (np.random.rand(q)<DNN.reseau[couche].sortie_entre_RBM(v))*1.
			v = (np.random.rand(p)<DNN.reseau[0].sortie_entre_RBM(v))*1.
		images.append(v)
		plt.imshow(v.reshape(20, 16),cmap='gray', vmin=0, vmax=1)
		plt.show()
	return images