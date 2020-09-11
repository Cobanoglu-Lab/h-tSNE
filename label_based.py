import pandas as pd
import numpy as np
import scipy.io
genes = pd.read_csv('hg19/genes.tsv', sep='\t',header=None)
barcodes = pd.read_csv('hg19/barcodes.tsv', sep='\t',header=None)
labels = pd.read_csv('hg19/zheng17_bulk_lables.txt', sep='\n',header=None)
matrix = scipy.io.mmread('hg19/matrix.mtx')
matrix = pd.DataFrame.sparse.from_spmatrix(matrix)

genes #genes.values[:,0]
genes.loc[genes[1] == 'IL7R']

labels = pd.read_csv('hg19/zheng17_bulk_lables.txt', sep='\n',header=None)
#labels.values[0][0]
pd.unique(labels[0])

#hlabels = ['Hematopoietic Stem Cells', 'Common Lymphoid Progenitor Cells','Common Myeloid Progenitor Cells', 
#          'TCells']
#hlabels = ['Hematopoietic Stem Cells', 'TCells']
hlabels = ['Lymphoid','Myeloid','T Cells','CD4 T','Conventional T','Effector/Memory T']
for label in pd.unique(labels[0]):
    hlabels.append(label)

labeldict = {}
cnt = 0
for label in hlabels:
    print(str(cnt) + " " + label);
    labeldict[label] = cnt; cnt+=1

import networkx as nx
import matplotlib.pyplot as plt
nuniq = len(hlabels)
ajmatrix = np.zeros(shape=(nuniq,nuniq))

ajmatrix[13] =[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # 13 to 0 and 1
ajmatrix[0] = [0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0]# 0 to 2 and 9
ajmatrix[1] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0] # 1 to 11 and 15
ajmatrix[2] = [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0] # 2 to 3 and 6
ajmatrix[3] = [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0] # 3 to 4 and 8
ajmatrix[4] = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0] # 4 to 14 5
ajmatrix[5] = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1] # to 16 and 10
ajmatrix[6] = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] # to 7

#ajmatrix[0] = [0,0,0,0,0,0,0,0,0,1,0,0,0] # 0 to 9
#ajmatrix[1] = [0,0,1,1,1,0,1,0,0,0,1,0,1] # 1 to 2,3,4,6,10,12
#ajmatrix[9] = [0,1,0,0,0,1,0,1,1,0,0,1,0] # 9 to 1,5,7,8,11

graph = nx.from_numpy_matrix(ajmatrix)

graph
plt.subplot(121)
glabels={}

glabels[0] = r'$0$'
glabels[1] = r'$1$'
glabels[2] = r'$2$'
glabels[3] = r'$3$'
glabels[4] = r'$4$'
glabels[5] = r'$5$'
glabels[6] = r'$6$'
glabels[7] = r'$7$'
glabels[8] = r'$8$'
glabels[9] = r'$9$'
glabels[10] = r'$10$'
glabels[11] = r'$11$'
glabels[12] = r'$12$'
glabels[13] = r'$13$'
glabels[14] = r'$14$'
glabels[15] = r'$15$'
glabels[16] = r'$16$'

#glabels[0]=r'$H$'
#glabels[1]=r'$T$'
#glabels[9]=r'$C$'
nx.draw(graph,labels=glabels)

#(nx.shortest_path_length(graph, 10, 12)-2)/3 
(nx.shortest_path_length(graph, 6, 7)-1)/8

# https://towardsdatascience.com/t-sne-python-example-1ded9953f26
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.manifold import _utils
from scipy.spatial.distance import is_valid_dm
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

MACHINE_EPSILON = np.finfo(np.double).eps # 1.0 + eps != 1.0
n_components = 2 # 2D
perplexity = 30  # Number of local neighbors.
N_ITER = 1000

#global ext_dist

def _joint_probabilities(distances, desired_perplexity, verbose):
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(distances, desired_perplexity, verbose)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def _gradient_descent(obj_func, p0, args, it=0, n_iter=N_ITER, n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it
    
    for i in range(it, n_iter):
        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        #print("[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f" % (i + 1, error, grad_norm))
        if error < best_error:
                best_error = error
                best_iter = i
        elif i - best_iter > n_iter_without_progress:
            print("Early stopping " + str(i))
            break

        if grad_norm <= min_grad_norm:
            print("Grad norm " + str(i))
            break
    return p

#def custom_pairwise(u,v):
#    return ((u-v)**2).sum() # sqeucledian
#    # distance to another node

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)
    
    # Probability dist over the points in the low-dim mapping
    # Degree of freedom of students t-dist. 
    #dist = pdist(X_embedded, "sqeuclidean")
    
    # START NEW DIST FUNCTION: KEVIN
    dist = pdist(X_embedded, 'sqeuclidean')
    #ext_dist = dist
    # END
    
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    
    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    
    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    return kl_divergence, grad

def path_dist(label1, label2):
    i = labeldict.get(labels.values[label1][0])
    j = labeldict.get(labels.values[label2][0])
    #return (nx.shortest_path_length(graph, i, j)-1)
    return (nx.shortest_path_length(graph, i, j)-1)/8 # -2 is common to all graphs so siblings have a zero weight 

def path_pairwise(x, factor = 1, squared=True):
    dists = np.zeros((x.shape[0], x.shape[1]))
    for i, row_x in enumerate(x):     # loops over rows of `x`
        for j, row_y in enumerate(x): # loops over rows of `y`
            #dists[i, j] = path_dist(i,j)*factor#np.sum((row_x - row_y)**2)# * path_dist(i,j)
            dists[i, j] = (1-factor)+path_dist(i,j)*factor
    return dists


def fit(X, xlabels,factor):
    X = X.transpose() # KEVIN
    n_samples = X.shape[0]
    
    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)
    pathpairwise = path_pairwise(X, factor)#.15
    np.fill_diagonal(pathpairwise, 0)
    distances = np.multiply(distances,pathpairwise)
    
    #norm distances
    distances=(distances-distances.min())/(distances.max()-distances.min())
    #distances=(distances-distances.mean())/distances.std()
    
    # Compute joint probabilities p_ij from distances.
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    
    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    # KEVIN: modify to include what we know as meaningful samples? (at nodes of tree).
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)
    
    degrees_of_freedom = max(n_components - 1, 1)
    
    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    params = X_embedded.ravel()
    obj_func = _kl_divergence
    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])
    X_embedded = params.reshape(n_samples, n_components)
    return X_embedded

import statistics
ncols = 6000
#ncols = 500
submatrix = matrix.iloc[:,0:ncols]
sums = submatrix.sum(axis=1)
rows = []
for _i in range(len(sums)):
    if sums[_i] <= 0.0:
        rows.append(_i)
    else:
        var = statistics.variance(submatrix.iloc[_i,:])
        if(var < 0.1): #.1
            rows.append(_i)
            
submatrix = submatrix.drop(rows)
submatrix

#submatrix = submatrix.iloc[0:submatrix.shape[1],:] # for small ncols
submatrix = submatrix.iloc[:,0:submatrix.shape[0]] # for large ncols'
submatrix.shape

ncols = submatrix.shape[1]

sums = submatrix.sum(axis=1)
med = statistics.median(sums) # median of row sums
#colsums = submatrix.sum(axis=0) # median of each cell (column)

submatrix = (submatrix.div(submatrix.sum(axis=0),axis=1)).multiply(med) # normalize cols by their sums & multiply by median of row sums
#submatrix = submatrix.div(submatrix.sum(axis=1),axis=0) # normalize rows by their sums
#print(submatrix.mean())
#print(submatrix.std())
submatrix=(submatrix-submatrix.mean())/submatrix.std() #https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame/48651066
submatrix

import sys
margs = sys.argv

ncols=submatrix.shape[0]
X_embedded = fit(submatrix, labels[0:ncols],float(margs[1]))

y = np.asarray(labels[0:ncols]).ravel()
#sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')

#SAVE:
#plt = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.get_figure().savefig(str(margs[2]) + '.png')

import pickle
pickle.dump(X_embedded,open('L6000'+str(margs[2]) + '.pickle','wb'))