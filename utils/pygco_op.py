from pygco import cut_from_graph
import numpy as np
import sys

class DiscreteEnergyMinimize:
    def __init__(self, nlabel, lamb, value1=100, value2=10000, niter = 10):
        '''
        Args:
            nlabel - int
            lamb - float
                should be positive
            value1 - int
                defaults to be 100
            value2 - int
                defaults to be 10000
        '''
        self.nlabel = nlabel
        self.lamb = lamb
        self.value1 = value1
        self.value2 = value2
        self.niter = niter
        self.pairwise_cost = -self.value1*self.lamb*np.eye(self.nlabel)
        self.pairwise_cost = self.pairwise_cost.astype(np.int32)

    def solve(self, unary_term, pairwise_term, k):
        '''
        Args :
            unary_term - Numpy 2d array [nvertex, nlabel]
                unary_term term to be minimized
            pairwise_term - Numpy 2d array [nvertex, nvertex]
                pairwise_term term to be minimized
            k - int
        '''
        assert unary_term.shape[1]==self.nlabel, "Unary term have wrong labels"
        nvertex = unary_term.shape[0]
        assert pairwise_term.shape==(nvertex, nvertex), "Pairwise term  haver wrong shape"
        
        unary_term = unary_term*self.value1*self.value2
        unary_term = unary_term.astype(np.int32)
        
        nedges = nvertex*(nvertex-1)/2
        nedges = int(nedges)
        self.edges = np.zeros([nedges, 3], dtype=np.float32)
        
        idx = 0
        for i in range(nvertex):
            for j in range(i+1, nvertex):
                self.edges[idx] = [i, j, -self.value2*pairwise_term[i][j]]
                idx+=1
 
        self.edges = self.edges.astype(np.int32)

        binary_vector = np.zeros([nvertex, self.nlabel], dtype=np.float32) 
        energy = 0
        keep = unary_term
        for _ in range(k):
            results = cut_from_graph(edges=self.edges, unary_cost=unary_term, pairwise_cost=self.pairwise_cost, n_iter=self.niter, algorithm='swap') 
            for i, j in enumerate(results): 
                binary_vector[i][j] = 1
                unary_term[i][j] = np.iinfo(np.int32).max//2

        return binary_vector
