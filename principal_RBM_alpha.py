import numpy as np
import scipy.io as scio
import numpy.random as rd

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

def generer_image_RBM(rbm, num_of_it, num_of_img):
    n = data['numclass'] # n is the population per class in the dataset
    k = n/num_of_img # we generate the same amount of images per class
    # let's generate k images per class :
    for _class in data['classlabels'] :
        for i in range(k):
            # gibbs sampling....
            pass
    return 0