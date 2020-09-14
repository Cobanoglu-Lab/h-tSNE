import networkx as nx
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.manifold import _utils
from scipy.spatial.distance import is_valid_dm

class HTSNE:
    """
        Base TSNE code is based on the below source & the sklearn.manifold implementation.
        ref: https://towardsdatascience.com/t-sne-python-example-1ded9953f26
    """
    def __init__(self, graph_labels, ajmatrix):
        self.MACHINE_EPSILON = np.finfo(np.double).eps # 1.0 + eps != 1.0
        self.n_components = 2 # 2D
        self.perplexity = 30  # Number of local neighbors.
        self.N_ITER = 1000
        self.X_embedded = None
        self.labels = None
        self.labeldict = {}
        self.graph = None
        self.max_shortestpath = 0
        self._init_graph(graph_labels, ajmatrix)
        
    def _init_graph(self, graph_labels, ajmatrix):
        # Dictionary from labels:
        cnt = 0
        for label in graph_labels:
            #print(str(cnt) + " " + label);
            self.labeldict[label] = cnt; cnt+=1
            
        # Make graph & find maximum shortest path:
        self.graph = nx.from_numpy_matrix(ajmatrix)
        for i in range(self.graph.size()):
            for j in range(self.graph.size()):
                path_len = nx.shortest_path_length(self.graph, i, j)
                if path_len > self.max_shortestpath:
                    self.max_shortestpath = path_len
                    
    def show_graph_info(self):
        cnt = 0; glabels={}
        for label in self.labeldict:
            glabels[cnt] = str(cnt)
            print("%d: %s" %(cnt, label)); cnt += 1
        nx.draw(self.graph, labels = glabels)
        
    def fit(self, X, xlabels, factor, random = True, n_iterations = None, transpose = True, X_embedded=None):
        if n_iterations == None:
            n_iterations = self.N_ITER
        self.N_ITER = n_iterations
        
        if transpose:
            X = X.transpose() # KEVIN
        n_samples = X.shape[0]
        
        self.labels = xlabels

        # Compute euclidean distance
        distances = pairwise_distances(X, metric='euclidean', squared=True)
        pathpairwise = self.path_pairwise(X, factor)#.15
        np.fill_diagonal(pathpairwise, 0)
        distances = np.multiply(distances,pathpairwise)

        # Normalize distances
        distances=(distances-distances.min())/(distances.max()-distances.min())
        #distances=(distances-distances.mean())/distances.std()

        # Compute joint probabilities p_ij from distances.
        P = self._joint_probabilities(distances=distances, desired_perplexity=self.perplexity, verbose=False)

        # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
        # KEVIN: modify to include what we know as meaningful samples? (at nodes of tree).
        if random == True:
            self.X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, self.n_components).astype(np.float32)
        else:
            self.X_embedded = X_embedded
            
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples, X_embedded=self.X_embedded)
        
    def path_dist(self, label1, label2):
        i = self.labeldict.get(self.labels.values[label1][0])
        j = self.labeldict.get(self.labels.values[label2][0])
        return (nx.shortest_path_length(self.graph, i, j)-1)/8 # -2 is common to all graphs so siblings have a zero weight 

    def path_pairwise(self, x, factor = 1, squared=True):
        dists = np.zeros((x.shape[0], x.shape[1]))
        for i, row_x in enumerate(x):     # loops over rows of `x`
            for j, row_y in enumerate(x): # loops over rows of `y`
                dists[i, j] = (1-factor)+self.path_dist(i,j)*factor
        return dists

    def _joint_probabilities(self, distances, desired_perplexity, verbose):
        # Compute conditional probabilities such that they approximately match the desired perplexity
        distances = distances.astype(np.float32, copy=False)
        conditional_P = _utils._binary_search_perplexity(distances, desired_perplexity, verbose)
        P = conditional_P + conditional_P.T
        sum_P = np.maximum(np.sum(P), self.MACHINE_EPSILON)
        P = np.maximum(squareform(P) / sum_P, self.MACHINE_EPSILON)
        return P

    def _gradient_descent(self, obj_func=None, p0=None, args=None, it=0, n_iter=None, n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):
        
        if n_iter == None:
            n_iter = self.N_ITER
        
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

    def _kl_divergence(self, params=None, P=None, degrees_of_freedom=None, n_samples=None, n_components = None):
        if n_components == None:
            n_components = self.n_components
        
        self.X_embedded = params.reshape(n_samples, n_components)

        dist = pdist(self.X_embedded, 'sqeuclidean')
        dist /= degrees_of_freedom
        dist += 1.
        dist **= (degrees_of_freedom + 1.0) / -2.0
        Q = np.maximum(dist / (2.0 * np.sum(dist)), self.MACHINE_EPSILON)

        # Kullback-Leibler divergence of P and Q
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, self.MACHINE_EPSILON) / Q))

        # Gradient: dC/dY
        grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
        PQd = squareform((P - Q) * dist)
        for i in range(n_samples):
            grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                             self.X_embedded[i] - self.X_embedded)
        grad = grad.ravel()
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad *= c
        return kl_divergence, grad

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded):
        params = X_embedded.ravel()
        obj_func = self._kl_divergence
        params = self._gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, self.n_components])
        X_embedded = params.reshape(n_samples, self.n_components)
        return X_embedded
