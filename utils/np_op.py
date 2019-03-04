from tqdm import tqdm

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)

import copy

def zero_padding2nmul(inputs, mul):
    '''Add zero padding to inputs to be multiple of mul
    Args:
        inputs - np array
        mul - int

    Return:
        np array (inputs + zero_padding)
        int original input size
    '''
    input_shape = list(inputs.shape)
    ndata = input_shape[0]
    if ndata%mul==0: return inputs, ndata
    input_shape[0] = mul-ndata%mul
    return np.concatenate([inputs, np.zeros(input_shape)], axis=0), ndata 

def greedy_max_matching_solver(array, plamb, k):
    '''
    Args:
        array - 2D Numpy array [nworkers, ntasks]
        plamb - float
        k - int
    Return:
        objective - 2D Numpy array [nworkers, ntasks]
            binary array 

    '''
    nworkers, ntasks = array.shape
    usage_count = np.zeros([ntasks])
    objective = np.zeros([nworkers, ntasks])
    array_copy = copy.copy(array)

    for _ in range(k):
        for w_idx in range(nworkers):
            tmp = -array_copy[w_idx]+plamb*(usage_count-objective[w_idx])
            t_idx = np.argmin(tmp)

            objective[w_idx][t_idx]=1
            usage_count[t_idx] = usage_count[t_idx]+1
            array_copy[w_idx][t_idx]=float('-inf')

    return objective

def greedy_max_matching_label_solver(up, plamb, labels):
    '''
    Args:
        up - 2D Numpy array [nworkers, ntasks]
        plamb - float
        labels - 1D Numpy array [nworkers]
    Return:
        objective - 2D Numpy array [nworkers, ntasks]
            binary array 
        value - float value of energy
    '''
    nworkers, ntasks = up.shape

    usage_count = np.zeros([ntasks])
    label_usage_count = dict()
    for w_idx in range(nworkers):
        w_label = labels[w_idx]
        if w_label in label_usage_count.keys():continue
        label_usage_count[w_label] = np.zeros([ntasks])
    objective = np.zeros([nworkers, ntasks])
    rand_order = np.random.permutation(nworkers)

    value = 0
    for w_idx in range(nworkers):
        sw_idx = rand_order[w_idx]
        w_label = labels[sw_idx]
        tmp = -up[sw_idx]+plamb*(usage_count-label_usage_count[w_label])
        t_idx = np.argmin(tmp) # greedy choice
        value += tmp[t_idx]

        objective[sw_idx][t_idx]=1
        label_usage_count[w_label][t_idx]=label_usage_count[w_label][t_idx]+1
        usage_count[t_idx] = usage_count[t_idx]+1

    return objective, value

def greedy_max_matching_label_solver_iter(up, plamb, labels, niter=1):
    '''
    Args:
        up - 2D Numpy array [nworkers, ntasks]
        plamb - float
        labels - 1D Numpy array [nworkers]
        niter - int, number of iterations
    Return:
        objective - 2D Numpy array [nworkers, ntasks]
            binary array 
    '''
    min_objective, min_value = greedy_max_matching_label_solver(up=up, plamb=plamb, labels=labels)
    for _ in range(1, niter):
        objective, value = greedy_max_matching_label_solver(up=up, plamb=plamb, labels=labels)
        if value < min_value:
            min_value = value
            min_objective = objective
    return min_objective

def greedy_max_matching_label_solver_k(array, plamb, labels, k):
    '''
    Args:
        array - 2D Numpy array [nworkers, ntasks]
        plamb - float
        labels - 1D Numpy array [nworkers]
        k - int
    Return:
        objective - 2D Numpy array [nworkers, ntasks]
            binary array 
        value - float value of energy
    '''
    nworkers, ntasks = array.shape

    usage_count = np.zeros([ntasks])
    label_usage_count = dict()
    for w_idx in range(nworkers):
        w_label = labels[w_idx]
        if w_label in label_usage_count.keys():continue
        label_usage_count[w_label] = np.zeros([ntasks])
    objective = np.zeros([nworkers, ntasks])
    rand_order = np.random.permutation(nworkers)

    value = 0

    array_copy = copy.copy(array)

    for _ in range(k):
        for w_idx in range(nworkers):
            sw_idx = rand_order[w_idx]
            w_label = labels[sw_idx]
            tmp = -array_copy[sw_idx]+plamb*(usage_count-label_usage_count[w_label])
            t_idx = np.argmin(tmp) # greedy choice
            value += tmp[t_idx]

            objective[sw_idx][t_idx]=1
            label_usage_count[w_label][t_idx]+=1
            usage_count[t_idx]+=1

            array_copy[sw_idx][t_idx]=float('-inf')

    return objective, value

def greedy_max_matching_label_solver_k_iter(array, plamb, labels, k, niter=1):
    '''
    Args:
        array - 2D Numpy array [nworkers, ntasks]
        plamb - float
        labels - 1D Numpy array [nworkers]
        k - int
        niter - int, number of iterations
    Return:
        objective - 2D Numpy array [nworkers, ntasks]
            binary array 
    '''
    min_objective, min_value = greedy_max_matching_label_solver_k(array=array, plamb=plamb, labels=labels, k=k)
    for _ in range(1, niter):
        objective, value = greedy_max_matching_label_solver_k(array=array, plamb=plamb, labels=labels, k=k)
        if value < min_value:
            min_value = value
            min_objective = objective
    return min_objective

def np_random_crop_4d(imgs, size):
    '''
    Args:
        imgs - 4d image NHWC
        size - list (rh, rw)
    '''
    rh, rw = size
    
    on, oh, ow, oc = imgs.shape

    cropped_imgs = np.zeros([on, rh, rw, oc])

    ch = np.random.randint(low=0, high=oh-rh, size=on)
    cw = np.random.randint(low=0, high=ow-rw, size=on)

    for idx in range(on):
        cropped_imgs[idx] = imgs[idx,ch[idx]:ch[idx]+rh,cw[idx]:cw[idx]+rw]
    
    return cropped_imgs

def np_center_crop_4d(imgs, size):
    '''
    Args:
        imgs - 4d image NHWC
        size - list (rh, rw)
    '''
    rh, rw = size
    
    on, oh, ow, oc = imgs.shape

    cropped_imgs = np.zeros([on, rh, rw, oc])

    ch = (oh-rh)//2
    cw = (ow-rw)//2

    for idx in range(on):
        cropped_imgs[idx] = imgs[idx,ch:ch+rh,cw:cw+rw]
    
    return cropped_imgs

def activate_k_1D(arr, k): 
    ''' activate top k-bit to 1 otherwise 0
    Dependency: numpy as np
    Args:
        arr - numpy 1D array
        k - int
    Return:
        arr_k_active - numpy 1D array
    '''
    length = len(arr)
    arr_k_active = np.zeros_like(arr)

    index = np.argsort(-arr)
    index = index[:k] # select highest k 

    arr_k_active[index] = 1
    return arr_k_active

def activate_k_2D(arr, k, session=None, batch_size=100):
    '''activate top k-bit to 1 otherwise 0
    Dependency: activate_k_1D, numpy as np
    Args:
        arr - numpy 2D array
        k - int
        session - session of tensorflow
            defaults to be None
        batch_size - int
            defautls to be 100
    Return 
        arr_k_active - numpy 2D array
    '''
    if session is None:
        arr_k_active = np.zeros_like(arr)
        ndata = len(arr)

        for i in range(ndata):
            arr_k_active[i] = activate_k_1D(arr=arr[i], k=k)

        return arr_k_active
    else:
        ndata, nfeature = arr.shape
        inputs = tf.placeholder(tf.float32, [batch_size, nfeature]) # [batch_size, nfeature]
        top_k = tf.nn.top_k(inputs, k=k+1)[0] # [batch_size, k+1]
        kth_element = 0.5*tf.add(top_k[:,k-1], top_k[:,k]) # [batch_size]
        kth_element = tf.reshape(kth_element, [-1,1]) # [batch_size, 1]
        k_hash_tensor = tf.cast(0.5*(tf.sign(tf.subtract(inputs, kth_element))+1), tf.int32) # [batch_size, nfeatures]
        
        if ndata%batch_size!=0:
            arr = np.concatenate([arr, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 
        nbatch=len(arr)//batch_size
        arr_k_active = list()
        for b in tqdm(range(nbatch), ascii=True, desc="batch"):
            feed_dict = {inputs : arr[b*batch_size:(b+1)*batch_size]}
            arr_k_active.append(session.run(k_hash_tensor, feed_dict=feed_dict))
        arr_k_active = np.concatenate(arr_k_active, axis=0)
        arr_k_active = arr_k_active[:ndata]
        return arr_k_active

def hamming_dist(arr1, arr2):
    '''hamming distance between two 1D arrays(arr1 and arr2)
    Dependency: numpy as np
    Args:
        arr1 - 1D numpy array with only 0 and 1
        arr2 - 1D numpy array with only 0 and 1
    Return:
        h_d - float hamming distance between arr1 and arr2
    '''
    assert len(arr1)==len(arr2), "Length of arr1 and arr2 should be same but %d and %d"%(len(arr1), len(arr2))

    length = len(arr1)
    h_d = 0
    for i in range(length):
        if arr1[i]!=arr2[i]:
            h_d+=1
    return h_d

def pairwise_distance_euclid_np(components, identity=0):
    '''Get the inter distances between components with metric
    Dependency: numpy as np
    Args:
        components - Numpy 2D array [ndata, nfeature]
        identity - float 
    Return:
        dist_matrix - 2D numpy array [n, n]
                        dist_matrix[i,j] = metric(components[i], components[j])
                        dist_matrix[i,i] = identity
    '''
    ndata, nfeature = components.shape
    dist_matrix = np.zeros([ndata, ndata])

    if ndata>1000:
        for i in tqdm(range(ndata), ascii=True, desc="idx"):
            for j in range(i, ndata):
                dist_matrix[i][j] = np.sum(np.square(components[i]-components[j]))
                dist_matrix[j][i] = dist_matrix[i][j]
    else:
        for i in range(ndata):
            for j in range(i, ndata):
                dist_matrix[i][j] = np.sum(np.square(components[i]-components[j]))
                dist_matrix[j][i] = dist_matrix[i][j]

    for i in range(ndata):
        dist_matrix[i][i] = identity
    return dist_matrix

def pairwise_dist(components, metric, identity=0):
    '''Get the inter distances between components with metric
    Dependency: numpy as np
    Args:
        components - list or numpy n sets
        metric - func(x1, x2) for the distance between x1 and x2
                 metric(a,b) = metric(b,a)
        identity - float 
    Return:
        dist_matrix - 2D numpy array [n, n]
                        dist_matrix[i,j] = metric(components[i], components[j])
                        dist_matrix[i,i] = identity
    '''
    ndata = len(components)
    dist_matrix = np.zeros([ndata, ndata])
    for i in range(ndata):
        for j in range(i,ndata):
            dist_matrix[i][j] = metric(components[i], components[j])
            dist_matrix[j][i] = dist_matrix[i][j]

    for i in range(ndata):
        dist_matrix[i][i] = identity
    return dist_matrix

def num2lscale(num, scale):
    '''Convert num to different scale
    Args:
        num - int
        scale - int
    Return:
        lscale - list
    '''
    x1 = num%scale
    x2 = num//scale
    if x2==0: return [num]

    lscale = num2lscale(x2, scale)
    lscale.append(x1)
    return lscale

def convert2list(n, base, fill):
    '''
    Args:
        n - int
        base -int
        fill -int
    Return:
        blist - list [fill]
    '''
    assert n<base**fill, "n should be bigger than base**fill"

    clist = list()
    for i in range(fill):
        clist.append(n%base)
        n = n//base
    clist.reverse()

    return clist

def plabel2subset(labels):
    '''
    Args :
        labels - Numpy 1D array or List
    Return :
        subsets - list of lists
    '''
    d = dict()
    nlabel = len(labels)
    for i in range(nlabel):
        l_value = labels[i]
        if l_value not in d.keys(): d[l_value] = list()
        d[l_value].append(i)

    subsets = list()
    for key in d.keys():
        if len(d[key])!=1: subsets.append(d[key])
    return subsets

def bws2label(objective, sparsity):
    '''binary with sparsity to label
    Args :
        objective - Numpy 2D array
            should be binary 0 or 1
        sparsity - int
            sparsity of objective
    Return :
        labels - Numpy 1D array
    '''
    ndata, embedding_dim = objective.shape
    labels = np.zeros([ndata])
    for i in range(ndata):
        idx = 0
        for j in range(embedding_dim):
            if objective[i][j]==1:
                labels[i]+=j*(embedding_dim**idx)
                idx+=1
        assert idx==sparsity, "There is an objective against sparsity condition"
    return labels

def clustering_by_two(X_samples): 
    '''
    Args:
        X_samples - Numpy 2D array [nsamples, nfeatures]
    Return:
        clustered_list - list ( nsamples//2 x [2] )
        labels - Numpy 1D array [nsamples]
    '''
    nsamples, nfeatures = X_samples.shape
    assert nsamples%2==0, "nsamples should be even"
    pairs = pairwise_distance_euclid_np(X_samples) # [nsamples, nsamples]
    max_value = np.max(pairs) + 1

    for i in range(nsamples): pairs[i][i] = max_value

    clustered_list = list()

    for _ in range(nsamples//2):
        closest = convert2list(np.argmin(pairs), nsamples, 2)
        for i in closest:
            for j in range(nsamples): pairs[i][j], pairs[j][i] = max_value, max_value
        clustered_list.append(closest)
    
    labels = np.zeros([nsamples])
    for i in range(nsamples): labels[i]=-1

    for idx in range(len(clustered_list)):
        pairs = clustered_list[idx]
        for i in pairs:
            labels[i] = idx

    return clustered_list, labels
    
def label2ordlabel(label):
    ndata = len(label)
    ordlabel = np.zeros([ndata], dtype=np.int32)
    old2new = dict()
    new_label = 0
    for idx in range(ndata):
        old_label = label[idx]
        if old_label not in old2new.keys():
            old2new[old_label]=new_label
            new_label+=1
        ordlabel[idx] = old2new[old_label]
    return ordlabel 

def get_power_array(power, length):
    tmp = 1
    array = list()
    for i in range(length):
        array.append(tmp)
        tmp*=power
    return array

