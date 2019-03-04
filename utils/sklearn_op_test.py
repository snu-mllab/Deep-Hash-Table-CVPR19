from sklearn_op import k_mean_labeling, KMeansClustering

import numpy as np

def test1():
    '''
    test for KMeansClustering

    Results -
        Fitting X_samples starts
        Fitting X_samples done
        [[ 1.  2.]
         [ 4.  1.]
         [ 4.  4.]]
        [[1 1 0]
         [1 0 1]
         [1 1 0]
         [0 1 1]
         [0 1 1]
         [1 1 0]]
    '''
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    kmc = KMeansClustering(X, 3)
    print(kmc.centers)
    sess = tf.Session()
    print(kmc.k_hash(X, sess))

def test2():
    '''
    test for k_mean_labeling

    Results -
        [0 0 0 1 1 1]
        [0 0 0 1 2 1]
        [1 3 1 0 0 2]
    '''
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    print(k_mean_labeling(X_samples=X, n_clusters=2))
    print(k_mean_labeling(X_samples=X, n_clusters=3))
    print(k_mean_labeling(X_samples=X, n_clusters=4))

if __name__=='__main__':
    test2()
