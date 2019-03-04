from eval_op import HashTableV2, get_srr_recall_precision_at_k_hash
from np_op import pairwise_distance_euclid_np
import numpy as np 

def HashTableV2_test():
    '''
    hash_key
    [[0 1 1]
     [0 1 1]
     [1 1 1]
     [1 1 0]
     [0 1 0]
     [0 0 0]
     [0 1 0]
     [1 1 0]
     [0 1 1]
     [1 1 0]]
    labelh
    [1 0 0 1 2 0 2 0 1 1]
    table
    [[5], [], [4, 6], [3, 7, 9], [], [], [0, 1, 8], [2]]
    h_dist_buckets0
    [0]
    h_dist_buckets1
    [1, 2, 4]
    h_dist_buckets2
    [3, 5, 6]
    h_dist_retriev0
    [5]
    h_dist_retriev1
    [4, 6]
    h_dist_retriev2
    [0, 1, 3, 7, 8, 9]
    '''
    np.random.seed(0) 

    nhash = 10
    nbit = 3
    nlabel = 3

    hash_key = np.array([np.random.choice(2, nbit, replace=True) for v in range(nhash)])
    labelh = np.random.choice(nlabel, nhash, replace=True)
    hashtable = HashTableV2(hash_key=hash_key, labelh=labelh, nlabel=nlabel)

    print("hash_key\n{}".format(hash_key))
    print("labelh\n{}".format(labelh))
    print("table\n{}".format(hashtable.table))
    for h_dist in range(nbit):
        print("h_dist_buckets{}\n{}".format(h_dist, hashtable.get_h_dist_buckets(hash_key[5], h_dist)))

    for h_dist in range(nbit):
        print("h_dist_retriev{}\n{}".format(h_dist, hashtable.get_retrieve_set(hash_key[5], h_dist)))

def get_srr_recall_precision_at_k_hash_test():
    np.random.seed(0) 

    nhash = 10
    nbit = 3
    nlabel = 3

    hash_key = np.array([np.random.choice(2, nbit, replace=True) for v in range(nhash)])
    hashdist_matrix = pairwise_distance_euclid_np(hash_key, identity=0)
    hashargsort_matrix = np.argsort(hashdist_matrix, axis=-1)

    print("hash_key\n{}".format(hash_key))
    print("hashdist_matrix\n{}".format(hashdist_matrix))
    print("hashargsort_matrix\n{}".format(hashargsort_matrix))
    k_set = [1,4,8]
    nk, k_max = len(k_set), max(k_set) 
        
    issame=True
    for q_idx in range(nhash):
        print("q_idx : {}".format(q_idx))
        argsort = hashargsort_matrix[q_idx]
        hashdist = hashdist_matrix[q_idx] 

        closest_idx = argsort[0]
        if closest_idx==q_idx and issame:
            closest_idx = argsort[1]
        closest_dist = hashdist[closest_idx]
        print("closest_dist : {}".format(closest_dist))
        print("closest_idx : {}".format(closest_idx))

        retrieve_set = list()
        idx = 0
        while idx<nhash:
            h_idx = argsort[idx]
            if hashdist[h_idx] > closest_dist: 
                closest_dist = hashdist[h_idx]
                break
            if h_idx==q_idx and issame:
                idx+=1
                continue
            idx+=1
            retrieve_set.append(h_idx)
        print("initial retrieve_set : {}".format(retrieve_set)) 

        for k_idx in range(nk):
            nretrieve = k_set[k_idx]
            while len(retrieve_set)<nretrieve:
                while idx<nhash:
                    h_idx = argsort[idx]
                    if hashdist[h_idx] > closest_dist: 
                        closest_dist = hashdist[h_idx]
                        break
                    if h_idx==q_idx and issame:
                        idx+=1
                        continue
                    idx+=1
                    retrieve_set.append(h_idx)
            print("retrieve_set {} :{}".format(k_idx, retrieve_set))

if __name__=='__main__':
    #HashTableV2_test()
    get_srr_recall_precision_at_k_hash_test()

