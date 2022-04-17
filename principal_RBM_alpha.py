import numpy as np
import scipy.io as scio
import numpy.random as rd
import matplotlib.pyplot as plt

data = scio.loadmat('binaryalphadigs.mat')

mat = data['dat']
datasize = mat.shape
sample_size = mat[0,0].shape
new_format_size = (datasize[0],datasize[1],sample_size[0]*sample_size[1])
# the new format is a flattened version of the raw samples

dataset = np.zeros(new_format_size)
for char in range(datasize[0]) :
    for samp in range(datasize[1]):
        dataset[char,samp] = mat[char,samp].flatten()

# 1-hot-enode of labels
datalabels = np.zeros((datasize[0],datasize[1], datasize[0]))
for i in range(datasize[0]):
    datalabels[i,:,i] = 1

def lire_alpha_digits(character_index_to_learn):
    return dataset[character_index_to_learn]

class RBM:
    def __init__(self, input_bias, output_bias ,weights):
        self.a = input_bias
        self.b = output_bias
        self.W = weights
    

def init_RBM(i,o): # i and o are input and ouptut sizes
    layer = RBM(np.zeros(i),np.zeros(o),rd.normal(loc = 0.0, scale = 0.1, size =(i,o) ))
    return layer

def sigmoid(x):
    return 1./(1+np.exp(-x))
    
def entree_sortie_RBM(rbm, input_data):
    return sigmoid(rbm.W.dot(input_data) + rbm.a)

def sortie_entree_RBM(rbm, output_data):
    return sigmoid(output_data.dot(rbm.W.T) + rbm.b.T)

def train_RBM(rbm, epochs, learning_rate, batch_size, data_size):
    
    for t in epochs :
        print("Epoch : ", t, "/", epochs)
        
        for batch in dataset[:,::int(data_size/batch_size)] :
            h = entree_sortie_RBM(rbm, batch)
            pos_grad = batch.dot(h)
            
            #Contrastive Divergence -1
            batch_prime = sortie_entree_RBM(rbm, h)
            h_prime = entree_sortie_RBM(rbm, batch_prime)
            

            # !!! : need to implement an avaergaing procedure before updating W, b and a
            neg_grad = batch_prime.dot(h_prime)
            rbm.W -= learning_rate*(pos_grad-neg_grad)
            rbm.a -= learning_rate*(batch - batch_prime)
            rbm.b -= learning_rate*(h - h_prime)
            
            err = np.linalg.norm(batch - batch_prime, 2)
            print(err)
            
    return rbm

def generer_image_RBM(rbm, n_it, n_img):
    images = []
    input_size = rbm.a.shape
    output_size = rbm.b.shape
    for i in range(n_img):
        v = (np.random.rand(input_size)<0.5).reshape(1,-1)
        for k in range(n_it) : # Gibbs sampling
            h = (np.random.rand(output_size)<entree_sortie_RBM(rbm, v*1.))*1.
            v = (np.random.rand(input_size)<sortie_entree_RBM(rbm, h*1.))*1.
        images.append(v)
        plt.imshow(v.reshape(20,16), cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()
    return images
        

