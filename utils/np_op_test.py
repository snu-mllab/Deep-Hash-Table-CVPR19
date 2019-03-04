import numpy as np
import tensorflow as tf
from np_op import activate_k_1D, activate_k_2D,\
                  pairwise_dist, hamming_dist, pairwise_distance_euclid_np,\
                  num2lscale, convert2list, plabel2subset,\
                  bws2label,\
                  clustering_by_two,\
                  label2ordlabel

def test1():
    '''
    test for activate_k_1D,
    test for activate_k_2D

    Results -
	arr : [3 6 5 4]
	activated arr : [0 1 1 0]
	arr : [[3 6 5 4]
	 [2 1 3 4]]
	activated arr : [[0 1 1 0]
	 [0 0 1 1]]
	activated arr : [[0 1 1 0]
	 [0 0 1 1]]
	arr :  [[3 6 5 4]
	 [2 1 3 4]]
    '''
    arr = np.array([3,6,5,4])
    print("arr :", arr)
    print("activated arr :", activate_k_1D(arr, 2))

    arr = np.array([[3,6,5,4],[2,1,3,4]])
    print("arr :", arr)
    print("activated arr :", activate_k_2D(arr, 2))
    
    with tf.Session() as sess:
        print("activated arr :", activate_k_2D(arr, 2, sess))
        print("arr : ", arr)

def test2():
    '''test for hamming_dist, pairwise_dist

    Results -
 	components :  [[1 0 1]
	 [0 1 1]
	 [1 1 1]]
	[[ 3.  2.  1.]
	 [ 2.  3.  1.]
	 [ 1.  1.  3.]]
    '''

    components = np.array([[1,0,1],[0,1,1],[1,1,1]])
    print("components : ",components)
    print(pairwise_dist(components, hamming_dist, identity=3))

def test3():
    '''test for num2lscale
    Results - 
    	num2lscale(82, 3) : [1, 0, 0, 0, 1]
	num2lscale(17, 2) : [1, 0, 0, 0, 1]
	num2lscale(17, 9) : [1, 8]
    '''
    print("num2lscale(82, 3) : {}".format(num2lscale(82, 3)))
    print("num2lscale(17, 2) : {}".format(num2lscale(17, 2)))
    print("num2lscale(17, 9) : {}".format(num2lscale(17, 9)))

def test4():
    '''test for convert2list
    Results -
	[0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
    '''
    print(convert2list(233, 2, 10))

def test5():
    '''test for plabel2subset
    Results -
 	labels : 
	[1 2 3 1 2 3]
	subsets : 
	[[0, 3], [1, 4], [2, 5]]
    '''
    label1 = [1,2,3]
    label2 = [1,2,3]
    labels = np.concatenate([label1, label2], axis=0)
    print("labels : \n{}".format(labels))
    print("subsets : \n{}".format(plabel2subset(labels)))

def test6():
    '''test for bws2label'''
    embedding_dim = 4
    ndata = 10
    sparsity = 3

    random_list = list()
    random_objective = np.zeros([ndata, embedding_dim])
    for i in range(ndata):
        tmp = np.random.choice(embedding_dim, sparsity, replace=False)
        random_list.append(tmp)
        for j in range(sparsity):
            random_objective[i][tmp[j]]=1

    print("random_list :\n{}".format(random_list))
    print("random_objective :\n{}".format(random_objective))
    print("label : \n{}".format(bws2label(objective=random_objective, sparsity=sparsity)))

def test7():
    '''test for clustering_by_two
    
    Results -
        clustered_list : 
        [[0, 1], [2, 5], [3, 4]]
        labels : 
        [ 0.  0.  1.  2.  2.  1.]
    '''
    X = np.array([[1, 2], [1, 3], [1, 0], [4, 3], [4, 4], [2, 0]])
    clustered_list, labels = clustering_by_two(X)
    print("clustered_list : \n{}".format(clustered_list))
    print("labels : \n{}".format(labels))
    print(np.concatenate(clustering_by_two(X)[0], axis=0))

def test8():
    '''debug for clustering_by_two
    Results - 
        pairs 
         : [[  0.   1.   4.  10.  13.   5.]
         [  1.   0.   9.   9.  10.  10.]
         [  4.   9.   0.  18.  25.   1.]
         [ 10.   9.  18.   0.   1.  13.]
         [ 13.  10.  25.   1.   0.  20.]
         [  5.  10.   1.  13.  20.   0.]]
        pairs 
         : [[ 26.   1.   4.  10.  13.   5.]
         [  1.  26.   9.   9.  10.  10.]
         [  4.   9.  26.  18.  25.   1.]
         [ 10.   9.  18.  26.   1.  13.]
         [ 13.  10.  25.   1.  26.  20.]
         [  5.  10.   1.  13.  20.  26.]]
        closest 
         : [0, 1]
        pairs 
         : [[ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  18.  25.   1.]
         [ 26.  26.  18.  26.   1.  13.]
         [ 26.  26.  25.   1.  26.  20.]
         [ 26.  26.   1.  13.  20.  26.]]
        closest 
         : [2, 5]
        pairs 
         : [[ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.   1.  26.]
         [ 26.  26.  26.   1.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]]
        closest 
         : [3, 4]
        pairs 
         : [[ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]
         [ 26.  26.  26.  26.  26.  26.]]
    '''

    X_samples = np.array([[1, 2], [1, 3], [1, 0], [4, 3], [4, 4], [2, 0]])
    nsamples, nfeatures = X_samples.shape
    assert nsamples%2==0, "nsamples should be even"
    pairs = pairwise_distance_euclid_np(X_samples) # [nsamples, nsamples]
    print("pairs \n : {}".format(pairs))
    max_value = np.max(pairs) + 1

    for i in range(nsamples): pairs[i][i] = max_value
    print("pairs \n : {}".format(pairs))

    clustered_list = list()

    for _ in range(nsamples//2):
        closest = convert2list(np.argmin(pairs), nsamples, 2)
        for i in closest:
            for j in range(nsamples): pairs[i][j], pairs[j][i] = max_value, max_value
        clustered_list.append(closest)
        print("closest \n : {}".format(closest))
        print("pairs \n : {}".format(pairs))
    
    labels = np.zeros([nsamples])
    for i in range(nsamples): labels[i]=-1

    for idx in range(len(clustered_list)):
        pairs = clustered_list[idx]
        for i in pairs:
            labels[i] = idx

def test9():
    '''test for bws2label(,sparsity=1)=argmax'''
    embedding_dim = 4
    ndata = 10
    sparsity = 1

    random_list = list()
    random_objective = np.zeros([ndata, embedding_dim])
    for i in range(ndata):
        tmp = np.random.choice(embedding_dim, sparsity, replace=False)
        random_list.append(tmp)
        for j in range(sparsity):
            random_objective[i][tmp[j]]=1

    print("random_list :\n{}".format(random_list))
    print("random_objective :\n{}".format(random_objective))
    print("label : \n{}".format(bws2label(objective=random_objective, sparsity=sparsity)))

def test10(): 
    '''test for label2ordlabel'''
    label = np.array([4,1,5,2,2,4,1,5])
    new_label = label2ordlabel(label)

    print("label : \n{}".format(label))
    print("new label : \n{}".format(new_label))

if __name__=='__main__':
    test10()



