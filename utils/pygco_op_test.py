import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.pygco_op import DiscreteEnergyMinimize

import numpy as np

def get_energy(unary_term, pairwise_term, lamb, binary_ans):
    value = 0
    nvertex, nlabel = unary_term.shape
    value += np.sum(np.multiply(unary_term, binary_ans))

    for i in range(nvertex):
        for j in range(i+1, nvertex):
            value += lamb*pairwise_term[i][j]*np.sum(np.multiply(binary_ans[i], binary_ans[j]))
    return value

def get_random_binary_vector(nvertex, nlabel, k):
    binary_vector = np.zeros([nvertex, nlabel], dtype=np.float32)
    for i in range(nvertex):
        pos = np.random.choice(nlabel, size=k, replace=False)
        binary_vector[i][pos]=1
    return binary_vector

def test1():
    print("============================test1============================")
    nvertex = 8
    nlabel = 6
    k = 4
    niter = 10000
    lamb = 0.1

    unary_potential = np.random.random([nvertex, nlabel])
    pairwise_term = np.ones([nvertex, nvertex])

    DEM = DiscreteEnergyMinimize(nlabel, lamb)
    binary_vector = DEM.solve(-unary_potential, pairwise_term, k)
    for i in range(nvertex):
        if np.sum(binary_vector[i])!=k:
            print("Not allowd alpha beta swapping")
            print(binary_vector)
            sys.exit()

    alpha_beta_energy = get_energy(-unary_potential, pairwise_term, lamb, binary_vector)
    print("alpha beta swappping : {}".format(binary_vector))
    print("Energy : {}".format(alpha_beta_energy))

    ncount = 0
    for i in range(niter):
        random_binary= get_random_binary_vector(nvertex, nlabel, k)
        random_energy = get_energy(-unary_potential, pairwise_term, lamb, random_binary)
        if random_energy<alpha_beta_energy: ncount+=1
    print("The number of cases that random is better than alpha beta swapping {}/{}".format(ncount, niter))

def test2():
    print("============================test2============================")
    nvertex = 2
    nlabel = 4
    k = 2
    lamb = 0

    unary_potential = np.random.random([nvertex, nlabel])
    pairwise_term = np.ones([nvertex, nvertex])

    DEM = DiscreteEnergyMinimize(nlabel, lamb)
    print("unary_potential\n : {}".format(unary_potential))
    binary_vector = DEM.solve(-unary_potential, pairwise_term, k)
    for i in range(nvertex):
        if np.sum(binary_vector[i])!=k:
            print("Not allowd alpha beta swapping")
            print(binary_vector)
            sys.exit()

    alpha_beta_energy = get_energy(-unary_potential, pairwise_term, lamb, binary_vector)
    print("alpha beta swappping : {}".format(binary_vector))
    print("Energy : {}".format(alpha_beta_energy))

    lamb = 0.1
    DEM = DiscreteEnergyMinimize(nlabel, lamb)
    print("unary_potential\n : {}".format(unary_potential))
    binary_vector = DEM.solve(-unary_potential, pairwise_term, k)

    for i in range(nvertex):
        if np.sum(binary_vector[i])!=k:
            print("Not allowd alpha beta swapping")
            print(binary_vector)
            sys.exit()

    alpha_beta_energy = get_energy(-unary_potential, pairwise_term, lamb, binary_vector)
    print("alpha beta swappping : {}".format(binary_vector))
    print("Energy : {}".format(alpha_beta_energy))

if __name__=='__main__':
    test1()
    test2()
