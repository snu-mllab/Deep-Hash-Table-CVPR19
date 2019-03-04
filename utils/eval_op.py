import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.sklearn_op import custom_roc_curve
from utils.np_op import convert2list
# .
import tensorflow as tf
import numpy as np

import itertools

def get_srr_recall_precision_at_k_hash(hashdist_matrix, hashargsort_matrix, dist_matrix, labelq, labelh, k_set, issame):
    '''
    Args:
        hashdist_matrix - Numpy 2d array [nquery, nhash]
            Hash distance
        hashargsort_matrix - Numpy 2d array [nquery, nhash]
            Hash argsort
        dist_matrix - Numpy 2d array [nquery, nhash]
            Distance from query from each data
        labelq - Numpy 1D array [nquery]
        labelh - Numpy 1D array [nhash]
        k_set - list (nk) increasing order 
        issame - bool
            True => if query and dist is same 
            False => otherwise
    Return:
        srr - Numpy 2D array [nquery] 
        recall_value_set - list [nk]
        precision_value_set - list [nk]
    '''
    nquery, nhash = dist_matrix.shape
    assert issame==(nquery==nhash), "Wrong issame parameter"

    nk, k_max = len(k_set), max(k_set) 
    recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
    srr = np.zeros([nk, nquery])

    for q_idx in range(nquery):
        argsort = hashargsort_matrix[q_idx]
        hashdist = hashdist_matrix[q_idx] 
        dist = dist_matrix[q_idx] 

        closest_idx = argsort[0]
        if closest_idx==q_idx and issame:
            closest_idx = argsort[1]

        closest_dist = hashdist[closest_idx]
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
                    retrieve_set.append(h_idx)
                    idx+=1
            srr[k_idx][q_idx] = len(retrieve_set)
            retrieve_set_sort = np.array(retrieve_set)
            retrieve_set_sort = retrieve_set_sort[np.argsort(dist[retrieve_set_sort])][:nretrieve]

            flag = 0
            for r_idx in retrieve_set_sort:
                if labelq[q_idx]==labelh[r_idx]:
                    precision_correct_set[k_idx]+=1
                    if flag==0:
                        recall_correct_set[k_idx]+=1
                        flag=1
    if issame:
        srr/=(nhash-1)
    else:
        srr/=(nhash)

    recall_value_set = list()
    precision_value_set = list()
    for k_idx in range(nk):
        recall_value_set.append(recall_correct_set[k_idx]/nquery)
        precision_value_set.append(precision_correct_set[k_idx]/nquery/k_set[k_idx])

    return srr, recall_value_set, precision_value_set

class HashTableV2:
    '''binary hash key'''
    def __init__(self, hash_key, labelh, nlabel):
        '''
        Args:
            hash_key - Numpy 2d array [nhash, nbit]
                Binary
            labelh - Numpy 1D array [nhash]
                the label of each hash key
            nlabel - the number of labels
        '''
        self.hash_key = hash_key
        self.nhash, self.nbit = self.hash_key.shape
        self.labelh = labelh
        self.nlabel = nlabel

        assert self.nhash==len(self.labelh), "Wrong label"

        self.nbucket = 2**self.nbit
        #================================================
        tmp = 1
        self.convertaux = np.zeros([self.nbit])
        for idx in range(self.nbit):
            self.convertaux[idx] = tmp
            tmp*=2
        #================================================
        self.h_dist2combinations = dict()
        for h_dist in range(1, self.nbit):
            self.h_dist2combinations[h_dist] = np.array(list(itertools.combinations(np.arange(self.nbit), h_dist)))
        #================================================
        self.hash_distribution = np.zeros([self.nbucket, self.nlabel]) 
        self.table = [list() for b_idx in range(self.nbucket)]
        for hash_idx in range(self.nhash):
            b_idx = self.key2bucket(self.hash_key[hash_idx])
            l_idx = self.labelh[hash_idx]
            self.hash_distribution[b_idx][l_idx]+=1
            self.table[b_idx].append(hash_idx)

    def key2bucket(self, key):
        '''key - Binary [nbit] array'''
        return int(np.sum(np.multiply(key, self.convertaux)))

    def get_h_dist_buckets(self, key, h_dist):
        if h_dist==0: return [self.key2bucket(key)]
        combinations = self.h_dist2combinations[h_dist]
        bucket_list = list()
        for combination in combinations:
            tmp = key.copy()
            for c_idx in combination:
                tmp[c_idx] = 1 - tmp[c_idx]
            bucket_list.append(self.key2bucket(tmp))
        return bucket_list

    def get_retrieve_set(self, q_key, h_dist):
        '''
        q_key -  [nbit] array 
        h_dist - int 
            hamming distance
        '''
        retrieve_set = set()

        for b_idx in self.get_h_dist_buckets(q_key, h_dist):
            retrieve_set |= set(self.table[b_idx])
        return retrieve_set

    def get_nmi(self):
        '''
        Return:
            nmi - int
        '''
        ncluster, nlabel = self.hash_distribution.shape
        cluster_array = np.sum(self.hash_distribution, axis=1)
        label_array = np.sum(self.hash_distribution, axis=0)

        ndata = np.sum(self.hash_distribution)

        cluster_prob = cluster_array/ndata
        cluster_entropy = 0
        for c_idx in range(ncluster):
            if cluster_prob[c_idx]!=0:
                cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

        label_prob = label_array/ndata
        label_entropy = 0
        for l_idx in range(nlabel):
            if label_prob[l_idx]!=0:
                label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

        mutual_information = 0
        for c_idx in range(ncluster):
            for l_idx in range(nlabel):
                if self.hash_distribution[c_idx][l_idx]!=0:
                    mutual_information += self.hash_distribution[c_idx][l_idx]/ndata*np.log2(ndata*self.hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
        norm_term = (cluster_entropy+label_entropy)/2
        return mutual_information/norm_term

    def get_srr_recall_precision_at_k_hash(self, dist_matrix, query_key, labelq, k_set, issame, h_dist):
        '''
        Args:
            dist_matrix - Numpy 2d array [nquery, nhash]
                Distance from query from each data
            query_key - Numpy 2d array [nquery, nbit] 
            labelq - Numpy 1D array [nquery]
            h_dist - int > 0
                retrieve buckets hamming distance < h_dist
            k_set - list (nk) increasing order 
            issame - bool
                True => if query and dist is same 
                False => otherwise
        Return:
            srr - Numpy 2D array [nquery] 
            recall_value_set - list [nk]
            precision_value_set - list [nk]
        '''
        nquery = len(query_key)
        assert query_key.shape==(nquery, self.nbit), "Wrong query key shape"
        assert dist_matrix.shape == (nquery, self.nhash), "Wrong dist matrix"
        assert issame==(nquery== self.nhash), "Wrong issame parameter"

        nk, k_max = len(k_set), max(k_set) 
        recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
        srr = np.zeros([nquery])

        for q_idx in range(nquery):
            q_key = query_key[q_idx]
            exclude_set = set([q_idx]) if issame else set() # exclude my self if issame 
            retrieve_set = set()
            for hd_idx in range(h_dist):
                retrieve_set |= self.get_retrieve_set(q_key, hd_idx)
            retrieve_set -= exclude_set
            hd_idx = h_dist
            flag= 0 

            for k_idx in range(nk):
                nretrieve = k_set[k_idx]
                while len(retrieve_set)<nretrieve:
                    retrieve_set |= self.get_retrieve_set(q_key, hd_idx)
                    retrieve_set -= exclude_set
                    hd_idx+=1

                if k_idx==0: srr[q_idx] = len(retrieve_set)

                retrieve_set_sort = np.array(list(retrieve_set))
                retrieve_set_sort = retrieve_set_sort[np.argsort(dist_matrix[q_idx][retrieve_set_sort])][:nretrieve]

                flag = 0
                for r_idx in retrieve_set_sort:
                    if labelq[q_idx]==self.labelh[r_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag==0:
                            recall_correct_set[k_idx]+=1
                            flag=1

        if issame:
            srr/=(self.nhash-1)
        else:
            srr/=(self.nhash)

        recall_value_set = list()
        precision_value_set = list()
        for k_idx in range(nk):
            recall_value_set.append(recall_correct_set[k_idx]/nquery)
            precision_value_set.append(precision_correct_set[k_idx]/nquery/k_set[k_idx])

        return srr, recall_value_set, precision_value_set

# HashTee insert index
class HashTree:
    def __init__(self, depth, width, max_k_idx, labelh, nlabel):
        '''
        Args:
            depth - int
                depth of hash tree
            width - int
                depth of hash tree
            max_k_idx - Numpy 1D array [nhash, sparsity]
            labelh - Numpy 1D array [nhash]
                the label of each hash key
            nlabel - the number of labels
        '''
        self.depth = depth # k
        self.width = width # d
        self.max_k_idx = max_k_idx
        self.labelh = labelh
        self.nlabel = nlabel

        self.nhash, self.sparsity = self.max_k_idx.shape 

        assert self.nhash==len(self.labelh), "Number of data doesn't match"

        self.nbucket = self.width**self.depth

        self.hash_distribution = np.zeros([self.nbucket, self.nlabel]) 
        for h_idx in range(self.nhash):
            for s_idx in range(self.sparsity):
                self.hash_distribution[self.max_k_idx[h_idx][s_idx]][self.labelh[h_idx]]+=1

        self.nmi = self.get_nmi()

        self.tree = list()
        for b_idx in range(self.nbucket): self.tree.append(list())
        
        for h_idx in range(self.nhash):
            for s_idx in range(self.sparsity):
                self.tree[self.max_k_idx[h_idx][s_idx]].append(h_idx)

        self.idx_convert = list()
        tmp = 1
        for i in range(self.depth):
            self.idx_convert.append(tmp)
            tmp*=self.width
        self.idx_convert = np.array(self.idx_convert)[::-1] # [d**(k-1),...,1]

    def get_nmi(self):
        '''
        Return:
            nmi - int
        '''
        ncluster, nlabel = self.hash_distribution.shape
        cluster_array = np.sum(self.hash_distribution, axis=1)
        label_array = np.sum(self.hash_distribution, axis=0)

        ndata = np.sum(self.hash_distribution)

        cluster_prob = cluster_array/ndata
        cluster_entropy = 0
        for c_idx in range(ncluster):
            if cluster_prob[c_idx]!=0:
                cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

        label_prob = label_array/ndata
        label_entropy = 0
        for l_idx in range(nlabel):
            if label_prob[l_idx]!=0:
                label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

        mutual_information = 0
        for c_idx in range(ncluster):
            for l_idx in range(nlabel):
                if self.hash_distribution[c_idx][l_idx]!=0:
                    mutual_information += self.hash_distribution[c_idx][l_idx]/ndata*np.log2(ndata*self.hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
        norm_term = (cluster_entropy+label_entropy)/2
        return mutual_information/norm_term

    def get_srr_recall_precision_at_k_hash(self, dist_matrix, query_tree_idx, query_max_k_idx, labelq, k_set, issame):
        '''
        Args:
            dist_matrix - Numpy 2d array [nquery, nhash]
                Distance from query from each data
            query_tree_idx - Numpy 3d array [nquery, depth, width]
            query_max_k_idx - Numpy 1d array [nquery, sparsity]
            labelq - Numpy 1D array [nquery]
            k_set - list (nk)
            issame - bool
                True => if query and dist is same 
                False => otherwise
        Return:
            srr - Numpy 2D array [nk, nquery] 
            recall_value_set - list [nk]
            precision_value_set - list [nk]
        '''
        nquery, sparsity_q = query_max_k_idx.shape
        assert query_tree_idx.shape==(nquery, self.depth, self.width), "Wrong query tree idx shape"
        assert nquery==len(labelq), "Wrong query max idx shape"
        assert issame==(nquery== self.nhash), "Wrong issame parameter"
        assert dist_matrix.shape == (nquery, self.nhash), "Wrong dist matrix"

        nk = len(k_set)
        recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
        srr = np.zeros([nk, nquery])

        for q_idx in range(nquery):
            tree_idx = query_tree_idx[q_idx] # [depth, width]
            for k_idx in range(nk):
                nretrieve = k_set[k_idx]
                retrieve_set = set()

                exclude_set = set([q_idx]) if issame else set() # exclude my self if issame 
                for s_idx in range(sparsity_q):
                    retrieve_set |= set(self.tree[query_max_k_idx[q_idx][s_idx]])
                retrieve_set -= exclude_set

                idx = sparsity_q
                while len(retrieve_set)<nretrieve:
                    idx_list = convert2list(n=idx, base=self.width, fill=self.depth) # [depth]
                    st_idx= np.sum(np.multiply(np.array([tree_idx[v][idx_list[v]] for v in range(self.depth)]), self.idx_convert))
                    retrieve_set |= set(self.tree[st_idx])
                    retrieve_set -= exclude_set
                    idx+=1

                retrieve_set = np.array(list(retrieve_set))
                srr[k_idx][q_idx] = len(retrieve_set)
                retrieve_set = retrieve_set[np.argsort(dist_matrix[q_idx][retrieve_set])][:nretrieve]
                flag = 0
                for r_idx in retrieve_set:
                    if labelq[q_idx]==self.labelh[r_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag==0:
                            recall_correct_set[k_idx]+=1
                            flag=1
        if issame:
            srr/=(self.nhash-1)
        else:
            srr/=(self.nhash)

        recall_value_set = list()
        precision_value_set = list()
        for k_idx in range(nk):
            recall_value_set.append(recall_correct_set[k_idx]/nquery)
            precision_value_set.append(precision_correct_set[k_idx]/nquery/k_set[k_idx])

        return srr, recall_value_set, precision_value_set

# HashTable insert index on the key
class HashTable:
    def __init__(self, hash_key, labelh, nlabel):
        '''
        Args:
            hash_key - Numpy 2d array [nhash, nbucket]
                should be binary 0 or 1
                Add index in the following hash key buckets
            labelh - Numpy 1D array [nhash]
                the label of each hash key
            nlabel - the number of labels
        '''
        self.hash_key = hash_key
        self.nhash, self.nbucket = self.hash_key.shape
        self.labelh = labelh
        self.nlabel = nlabel

        assert self.nhash==len(self.labelh), "Wrong label"

        self.table = list()
        self.hash_count = np.zeros(nlabel)
        for hash_idx in range(self.nhash): 
            self.hash_count[self.labelh[hash_idx]]+=1

        self.hash_distribution = np.zeros([self.nbucket, self.nlabel]) 
        for hash_idx in range(self.nhash):
            for b_idx in range(self.nbucket):
                if self.hash_key[hash_idx][b_idx]==1:
                    self.hash_distribution[b_idx][self.labelh[hash_idx]]+=1


        for b_idx in range(self.nbucket):
            self.table.append(list())

        for hash_idx in range(self.nhash):
            for b_idx in range(self.nbucket):
                if self.hash_key[hash_idx][b_idx]==1:
                    self.table[b_idx].append(hash_idx)

    def get_retrieve_set(self, query_key):
        '''
        Args:
            query_key - Numpy 1D array
        '''
        assert query_key.ndim==1, "Query key dimension should be 1"
        retrieve_set = set()
        for idx in range(len(query_key)):
            if query_key[idx]==1:
                retrieve_set |= set(self.table[idx])
        return list(retrieve_set)

    def get_nmi(self):
        '''
        Return:
            nmi - int
        '''
        ncluster, nlabel = self.hash_distribution.shape
        cluster_array = np.sum(self.hash_distribution, axis=1)
        label_array = np.sum(self.hash_distribution, axis=0)

        ndata = np.sum(self.hash_distribution)

        cluster_prob = cluster_array/ndata
        cluster_entropy = 0
        for c_idx in range(ncluster):
            if cluster_prob[c_idx]!=0:
                cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

        label_prob = label_array/ndata
        label_entropy = 0
        for l_idx in range(nlabel):
            if label_prob[l_idx]!=0:
                label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

        mutual_information = 0
        for c_idx in range(ncluster):
            for l_idx in range(nlabel):
                if self.hash_distribution[c_idx][l_idx]!=0:
                    mutual_information += self.hash_distribution[c_idx][l_idx]/ndata*np.log2(ndata*self.hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
        norm_term = (cluster_entropy+label_entropy)/2
        return mutual_information/norm_term

    def get_std(self):
        cluster_array = np.sum(self.hash_distribution, axis=1)
        return np.std(cluster_array/self.nhash)

    def get_entropy(self):
        cluster_array = np.sum(self.hash_distribution, axis=1)
        cluster_prob = cluster_array/self.nhash
        cluster_entropy = 0
        for c_idx in range(ncluster):
            if cluster_prob[c_idx]!=0:
                cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])
        return cluster_entropy

    def get_recall_precision_IR(self, query_key, labelq, issame): 
        '''
        Args:
            query_key - Numpy 2d array [nquery, nbucket] 
                    shoulde be binary
            labelq - Numpy 1D array [nquery]
            issame - bool
                True => if query and dist is same 
                False => otherwise
        Return:
            recall_value
            precision_value
        '''
        nquery = len(query_key)
        assert query_key.shape==(nquery, self.nbucket), "Wrong query key shape"
        assert issame==(nquery== self.nhash), "Wrong issame parameter"

        recall_value = 0
        precision_value = 0

        for q_idx in range(nquery):
            q_label = labelq[q_idx]
            retrieve_set = set()

            exclude_set = set([q_idx]) if issame else set() # exclude my self if issame 
            for b_idx in range(self.nbucket):
                if query_key[q_idx][b_idx]==1:
                    retrieve_set|=set(self.table[b_idx])
            retrieve_set -= exclude_set
            retrieve_set = np.array(list(retrieve_set))
            nretrieve = len(retrieve_set)

            count = 0
            for r_idx in retrieve_set:
                if q_label==self.labelh[r_idx]:
                    count+=1
            if issame:
                recall_value+=count/(self.hash_count[q_label]-1)
            else:
                recall_value+=count/self.hash_count[q_label]
            if count!=0: 
                precision_value+=count/nretrieve

        recall_value/=nquery
        precision_value/=nquery
        return recall_value, precision_value

    def get_srr_recall_precision_at_k_hash(self, dist_matrix, query_key, labelq, base_activate_k, k_set, issame):
        '''
        Args:
            dist_matrix - Numpy 2d array [nquery, nhash]
                Distance from query from each data
            query_key - Numpy 2d array [nquery, nbucket] 
            labelq - Numpy 1D array [nquery]
            base_activate_k - int
            k_set - list (nk)
            issame - bool
                True => if query and dist is same 
                False => otherwise
        Return:
            srr - Numpy 2D array [nk, nquery] 
            recall_value_set - list [nk]
            precision_value_set - list [nk]
        '''
        nquery = len(query_key)
        assert query_key.shape==(nquery, self.nbucket), "Wrong query key shape"
        assert dist_matrix.shape == (nquery, self.nhash), "Wrong dist matrix"
        assert issame==(nquery== self.nhash), "Wrong issame parameter"

        nk, k_max = len(k_set), max(k_set) 
        recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
        srr = np.zeros([nk, nquery])

        for q_idx in range(nquery):
            for k_idx in range(nk):
                nretrieve = k_set[k_idx]
                query_sort = np.argsort(-query_key[q_idx])
                retrieve_set = set()

                exclude_set = set([q_idx]) if issame else set() # exclude my self if issame 
                for idx in range(base_activate_k):
                    retrieve_set |= set(self.table[query_sort[idx]])
                retrieve_set -= exclude_set

                idx = base_activate_k
                while len(retrieve_set)<nretrieve:
                    retrieve_set |= set(self.table[query_sort[idx]])
                    retrieve_set -= exclude_set
                    idx+=1 
                retrieve_set = np.array(list(retrieve_set))
                srr[k_idx][q_idx] = len(retrieve_set)
                retrieve_set = retrieve_set[np.argsort(dist_matrix[q_idx][retrieve_set])][:nretrieve]
                flag = 0
                for r_idx in retrieve_set:
                    if labelq[q_idx]==self.labelh[r_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag==0:
                            recall_correct_set[k_idx]+=1
                            flag=1
        if issame: srr/=(self.nhash-1)
        else: srr/=(self.nhash)

        recall_value_set = list()
        precision_value_set = list()
        for k_idx in range(nk):
            recall_value_set.append(recall_correct_set[k_idx]/nquery)
            precision_value_set.append(precision_correct_set[k_idx]/nquery/k_set[k_idx])

        return srr, recall_value_set, precision_value_set


def get_recall_precision_at_k_arg(arg_matrix, labelq, labelh, k_set, issame=False):
    '''Get recall value with
    search space and query is different
    Dependency: numpy as np
    Args:
        arg_matrix - 2D numpy array [nquery, nhash]
        labelq - 1D numpy array [nquery]
        labelh - 1D numpy array [nhash]
        k_set - list of int which is the k value for recall [nk]
        issame - bool determine wheter two matrix and same
    Return:
        recall_value_set - list of float with length k_set
        precision_value_set - list of float with length k_set
    '''
    nquery, nhash = len(labelq), len(labelh)
    nk, k_max  = len(k_set), max(k_set) 

    assert arg_matrix.shape == (nquery, nhash), "Wrong arg_matrix shape arg_matrix shape : {}, and labelq({}), labelh({})".format(arg_matrix.shape, label.shape, labelh.shape)
    assert issame==(nquery==nhash), "label should be same"

    recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
    for idx1 in range(nquery):
        count = 0 # prevent useless counting
        idx_close_from_i = arg_matrix[idx1]
        flag_set = np.zeros(nk) # for recall double counting for recall
        for idx2 in idx_close_from_i:
            if issame and idx2==idx1:
                continue #if data1, and data2 is same, exclude same idx
            count+=1
            if labelq[idx1]==labelh[idx2]:
                for k_idx in range(nk):
                    if count<=k_set[k_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag_set[k_idx]==0:
                            recall_correct_set[k_idx]+=1
                            flag_set[k_idx]=1
            if count>=k_max:
                break

    recall_value_set = list()
    precision_value_set = list()
    for k_idx in range(nk):
        k = k_set[k_idx]
        recall_value_set.append(recall_correct_set[k_idx]/nquery)
        precision_value_set.append(precision_correct_set[k_idx]/nquery/k)
    return recall_value_set, precision_value_set 

def get_recall_precision_at_k(dist_matrix, labelq, labelh, k_set, issame=False):
    '''Get recall value with
    search space and query is different
    Dependency: numpy as np
    Args:
        dist_matrix - 2D numpy array [nquery, nhash]
        labelq - 1D numpy array [nquery]
        labelh - 1D numpy array [nhash]
        k_set - list of int which is the k value for recall [nk]
        issame - bool determine wheter two matrix and same
    Return:
        recall_value_set - list of float with length k_set
        precision_value_set - list of float with length k_set
    '''
    nquery, nhash = len(labelq), len(labelh)
    nk, k_max  = len(k_set), max(k_set) 

    assert dist_matrix.shape == (nquery, nhash), "Wrong dist_matrix shape dist_matrix shape : {}, and labelq({}), labelh({})".format(dist_matrix.shape, label.shape, labelh.shape)
    assert issame==(nquery==nhash), "label should be same"

    recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
    for idx1 in range(nquery):
        count = 0 # prevent useless counting
        idx_close_from_i = np.argsort(dist_matrix[idx1])
        flag_set = np.zeros(nk) # for recall double counting for recall
        for idx2 in idx_close_from_i:
            if issame and idx2==idx1:
                continue #if data1, and data2 is same, exclude same idx
            count+=1
            if labelq[idx1]==labelh[idx2]:
                for k_idx in range(nk):
                    if count<=k_set[k_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag_set[k_idx]==0:
                            recall_correct_set[k_idx]+=1
                            flag_set[k_idx]=1
            if count>=k_max:
                break

    recall_value_set = list()
    precision_value_set = list()
    for k_idx in range(nk):
        k = k_set[k_idx]
        recall_value_set.append(recall_correct_set[k_idx]/nquery)
        precision_value_set.append(precision_correct_set[k_idx]/nquery/k)
    return recall_value_set, precision_value_set 

def get_nmi_suf_quick(index_array, label_array, ncluster, nlabel):
    '''
    Args:
        index_array - [ndata, k]
                value should be [0, ncluster)
        label_array - [ndata]
                value should be [0, nlabel)
        ncluster -  int
        nlabel - int
    Return:
        nmi - float
        suf - float
    '''
    if index_array.ndim==1:
        index_array = np.expand_dims(index_array, axis=-1)
    ndata, k_value = index_array.shape
    hash_distribution = np.zeros([ncluster, nlabel])
    
    for idx1 in range(ndata):
        tmp_l = label_array[idx1]
        for idx2 in range(k_value):
            hash_distribution[index_array[idx1][idx2]][tmp_l]+=1

    cluster_array = np.sum(hash_distribution, axis=1) # [ncluster]
    label_array = np.sum(hash_distribution, axis=0) # [nlabel]

    total_size = ndata*k_value

    cluster_prob = cluster_array/total_size
    cluster_entropy = 0
    for c_idx in range(ncluster):
        if cluster_prob[c_idx]!=0:
            cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

    label_prob = label_array/total_size
    label_entropy = 0
    for l_idx in range(nlabel):
        if label_prob[l_idx]!=0:
            label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

    mutual_information = 0
    for c_idx in range(ncluster):
        for l_idx in range(nlabel):
            if hash_distribution[c_idx][l_idx]!=0:
                mutual_information += hash_distribution[c_idx][l_idx]/total_size*np.log2(total_size*hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
    norm_term = (cluster_entropy+label_entropy)/2
    nmi = mutual_information/norm_term

    suf = np.sum(cluster_array)/np.sum(np.square(cluster_array))
    suf *= ndata
    suf /= k_value
    return nmi, suf

def get_nmi_quick(index_array, label_array, ncluster, nlabel):
    '''
    Args:
        index_array - [ndata, k]
                value should be [0, ncluster)
        label_array - [ndata]
                value should be [0, nlabel)
        ncluster -  int
        nlabel - int
    Return:
        nmi - float
    '''
    if index_array.ndim==1:
        index_array = np.expand_dims(index_array, axis=-1)
    ndata, k_value = index_array.shape
    hash_distribution = np.zeros([ncluster, nlabel])
    
    for idx1 in range(ndata):
        tmp_l = label_array[idx1]
        for idx2 in range(k_value):
            hash_distribution[index_array[idx1][idx2]][tmp_l]+=1

    cluster_array = np.sum(hash_distribution, axis=1)
    label_array = np.sum(hash_distribution, axis=0)

    total_size = ndata*k_value

    cluster_prob = cluster_array/total_size
    cluster_entropy = 0
    for c_idx in range(ncluster):
        if cluster_prob[c_idx]!=0:
            cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

    label_prob = label_array/total_size
    label_entropy = 0
    for l_idx in range(nlabel):
        if label_prob[l_idx]!=0:
            label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

    mutual_information = 0
    for c_idx in range(ncluster):
        for l_idx in range(nlabel):
            if hash_distribution[c_idx][l_idx]!=0:
                mutual_information += hash_distribution[c_idx][l_idx]/total_size*np.log2(total_size*hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
    norm_term = (cluster_entropy+label_entropy)/2
    return mutual_information/norm_term

def get_VAL_at_FAR(dist_matrix, labels, value):
    '''
    Args:
        dist_matrix - Numpy 1D array [npair]
            the distance between each pairs
        labels - Numpy 1D array [npair]
            if two pairs are same to be 1, 
                otherwise 0
        value - float
            get TPR at FAR==value

    Return:
        TPR at FAR==value
    '''
    labels = np.array(labels)
    dist_matrix = np.array(dist_matrix)
    npair = len(labels)
    fpr, tpr, _ = custom_roc_curve(labels=labels, distances=dist_matrix)

    assert value<1, "value should be less than 1"
    
    idx = 0
    while True:
        if fpr[idx]>value:
            break
        idx+=1
    return tpr[idx]

def get_VAL_at_FAR2(dist_matrix, labels, value, avoid_value):
    '''
    Args:
        dist_matrix - Numpy 1D array [npair]
            the distance between each pairs
        labels - Numpy 1D array [npair]
            if two pairs are same to be 1, 
                otherwise 0
        value - float
            get TPR at FAR==value
        avoid_value - float
            threshold that should be lower than avoid_value

    Return:
        TPR at FAR==value
    '''
    labels = np.array(labels)
    dist_matrix = np.array(dist_matrix)
    npair = len(labels)
    fpr, tpr, thr = custom_roc_curve(labels=labels, distances=dist_matrix)

    assert value<1, "value should be less than 1"
    
    idx = 0
    while True:
        if fpr[idx]>value:
            break
        idx+=1
    while idx>=0:
        if thr[idx]<avoid_value:
            break
        idx-=1
    return tpr[idx]

def get_DIR_at_FAR(label_gallery, label_genuine, label_impostor, dist_gg, dist_ig, value, k):
    '''
    Args:
        label_gallery - Numpy 1D array [ngallery]
            label of each gallery
        label_genuine - Numpy 1D array [ngenuine]
            label of each genuine probe
        label_impostor - Numpy 1D array [nimpostor]
            label of each impostor probe
        dist_gg - Numpy 2D array [ngenuine, ngallery]
            distance from each genuine probe to each gallery
        dist_ig - Numpy 2D array [nimpostor, ngallery]
            distance from impostor probe to each gallery
        k - int
        value - float
    Return:
         
    '''
    ngallery = len(label_gallery)
    ngenuine = len(label_genuine)
    nimpostor = len(label_impostor)

    assert dist_gg.shape==(ngenuine, ngallery), "dist_gg : wrong shape"
    assert dist_ig.shape==(nimpostor, ngallery), "dist_gg : wrong shape"

    thresholds = np.min(dist_ig, axis=-1) # [nimpostor]
    tau = sorted(thresholds)[int(nimpostor*value)]

    nsuccess = 0
    for gp_idx in range(ngenuine):
        g_sort = np.argsort(dist_gg[gp_idx])[:k]
        for g_idx in g_sort:
            if dist_gg[gp_idx][g_idx]>tau:
                break;
            if label_gallery[g_idx]==label_genuine[gp_idx]:
                nsuccess+=1
                break 

    dir_at_far = nsuccess/ngenuine
    return dir_at_far

def get_DIR_at_FAR2(label_gallery, label_genuine, label_impostor, dist_gg, dist_ig, value, k, avoid_value):
    '''
    Args:
        label_gallery - Numpy 1D array [ngallery]
            label of each gallery
        label_genuine - Numpy 1D array [ngenuine]
            label of each genuine probe
        label_impostor - Numpy 1D array [nimpostor]
            label of each impostor probe
        dist_gg - Numpy 2D array [ngenuine, ngallery]
            distance from each genuine probe to each gallery
        dist_ig - Numpy 2D array [nimpostor, ngallery]
            distance from impostor probe to each gallery
        k - int
        value - float
        avoid_value - float
    Return:
         
    '''
    ngallery = len(label_gallery)
    ngenuine = len(label_genuine)
    nimpostor = len(label_impostor)

    assert dist_gg.shape==(ngenuine, ngallery), "dist_gg : wrong shape"
    assert dist_ig.shape==(nimpostor, ngallery), "dist_gg : wrong shape"

    thresholds = np.min(dist_ig, axis=-1) # [nimpostor]
    thresholds_sorted = sorted(thresholds)

    tau = -1 
    for idx in range(int(nimpostor*value),-1,-1):
        if thresholds_sorted[idx]<avoid_value:
            tau = thresholds_sorted[idx] 
            break

    nsuccess = 0
    for gp_idx in range(ngenuine):
        g_sort = np.argsort(dist_gg[gp_idx])[:k]
        for g_idx in g_sort:
            if dist_gg[gp_idx][g_idx]>tau:
                break;
            if label_gallery[g_idx]==label_genuine[gp_idx]:
                nsuccess+=1
                break 

    dir_at_far = nsuccess/ngenuine
    return dir_at_far

def get_rank_k(label_gallery, label_probe, dist_pg, k):
    '''
    Args:
        label_gallery - Numpy 1D array [ngallery]
            label of each gallery
        label_probe - Numpy 1D array [nprobe]
            label of each probe
        dist_pg - Numpy 2D array [nprobe, ngallery]
            distance from probe to gallery set
        k - int
            the k in ran k
    Return:
        rank_k - float
    '''
    ngallery = len(label_gallery)
    nprobe = len(label_probe)

    nsuccess= 0
    for p_idx in range(nprobe):
        g_sort= np.argsort(dist_pg[p_idx])[:k] # small to big
        for g_idx in g_sort:
            if label_gallery[g_idx]==label_probe[p_idx]:
                nsuccess+=1
                break

    rank_k = nsuccess/nprobe
    return rank_k

if __name__ == '__main__':
    print(get_VAL_at_FAR([0.9, 0.6, 0.65, 0.2], [0,0, 1,1], 0.1))

