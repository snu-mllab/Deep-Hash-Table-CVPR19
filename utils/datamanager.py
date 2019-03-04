import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.np_op import np_random_crop_4d, clustering_by_two

import numpy as np
import random

class BasicDatamanager(object):
    def __init__(self, image, label, nclass):
        self.image = image
        self.label = label
        self.nclass = nclass
        self.ndata = len(self.label)

        self.fullidx = np.arange(self.ndata)
        self.start = 0
        self.end = 0

    def print_shape(self):
        print("Image shape : {}({})".format(self.image.shape, self.image.dtype))
        print("Label shape : {}({})".format(self.label.shape, self.label.dtype))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata):
            counter[int(self.label)]+=1
        return counter

    def next_batch(self, batch_size):
        '''
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            image
            label
        '''
        if self.start == 0 and self.end ==0:
            np.random.shuffle(self.fullidx) # shuffle first
        
        if self.end + batch_size > self.ndata:
            self.start = self.end
            self.end = (self.end + batch_size)%self.ndata
            self.subidx = np.append(self.fullidx[self.start:self.ndata], self.fullidx[0:self.end])
            self.start = 0
            self.end = 0
        else:
            self.start = self.end
            self.end += batch_size
            self.subidx = self.fullidx[self.start:self.end]

        return self.image[self.subidx], self.label[self.subidx].astype('int32')

class ContrastDatamanager(object):
    def __init__(self, image, label, nclass):
        '''
        Args:
            image -
            label - 
            nclass - number of total classes
        '''
        self.image = image
        self.label = label
        self.nclass = nclass

        self.ndata = len(self.label)

        self.pos_label_idx_set = [[idx for idx in range(self.ndata) if self.label[idx]==class_idx] for class_idx in range(self.nclass)]
        self.neg_label_idx_set = [[idx for idx in range(self.ndata) if self.label[idx]!=class_idx] for class_idx in range(self.nclass)]

        self.fullidx = np.arange(self.ndata)
        self.start = 0
        self.end = 0

    def print_shape(self):
        print("Image shape : {}".format(self.image.shape))
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata):
            counter[int(self.label)]+=1
        return counter

    def change_nsclass(self, value):
        self.nsclass = value

    def next_batch(self, batch_size):
        '''
        Make batch data which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            image - batch_size image
            image_pair - batch_size image
            label - batch_size binary label 
                    pos to be 1
                    neg to be 0
        '''
        if self.start==0 and self.end==0:
            np.random.shuffle(self.fullidx) # shuffle first
       
        npositive = batch_size//2
        nnegative = batch_size-npositive
        self.binary_label = np.append(np.ones(npositive), np.zeros(nnegative))
        
        if self.end + batch_size > self.ndata:
            self.start = self.end
            self.end = (self.end + batch_size)%self.ndata
            self.subidx = np.append(self.fullidx[self.start:self.ndata], self.fullidx[0:self.end])
            self.start = 0
            self.end = 0
        else:
            self.start = self.end
            self.end += batch_size
            self.subidx = self.fullidx[self.start:self.end]

        self.subidx_pair = list()
        for i in range(batch_size):
            anc_label = self.label[self.subidx[i]]
            if i < npositive:
                while True:
                    pos_sample = random.sample(self.pos_label_idx_set[anc_label], 1)[0]
                    if pos_sample!=self.subidx[i]:
                        break
                self.subidx_pair.append(pos_sample)
            else:
                self.subidx_pair.append(random.sample(self.neg_label_idx_set[anc_label], 1)[0])

        self.subidx_pair = np.array(self.subidx_pair)
        return self.image[self.subidx], self.image[self.subidx_pair], self.binary_label

class TripletDatamanager(object):
    def __init__(self, image, label, nclass, nsclass=2):
        '''
        Args:
            image -
            label - 
            nclass - number of total classes
            nsclass - When we select the batch, the number of classes it contain
        '''
        self.image = image
        self.label = label
        self.nclass = nclass
        self.nsclass = nsclass 

        self.ndata = len(self.label)

        # list with [self.nclass] each element = idx set of which label is cls_idx
        # initialize
        
        self.label_idx_set = list()
        for cls_idx in range(self.nclass): self.label_idx_set.append(list())
        # append
        for d_idx in range(self.ndata): self.label_idx_set[self.label[d_idx]].append(d_idx)
        # to numpy
        for cls_idx in range(self.nclass): self.label_idx_set[cls_idx] = np.array(self.label_idx_set[cls_idx])
        
        self.valid_class_set = [cls_idx for cls_idx in range(self.nclass) if len(self.label_idx_set[cls_idx])>1]
        self.ndata_idx = np.array([len(v) for v in self.label_idx_set])
        self.fullidx = [np.arange(self.ndata_idx[cls_idx], dtype=np.int32) for cls_idx in range(self.nclass)]
        self.start = np.zeros(self.nclass, dtype=np.int32)
        self.end = np.zeros(self.nclass, dtype=np.int32)

    def print_shape(self):
        print("Image shape : {}".format(self.image.shape))
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata): counter[int(self.label)]+=1
        return counter

    def change_nsclass(self, value):
        self.nsclass = value

    def add_embed(self, embed):
        self.embed = embed

    def next_batch_with_embed(self, batch_size):
        '''
        Make batch data which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            image - batch_size image
            label - batch_size label
        '''
        assert batch_size%self.nsclass == 0, "Batchsize(%d) should be divided by nsclass(%d)"%(batch_size, self.nsclass)

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first
        
        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class

                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        # self.subidx [batch_per_class] x nsclass
        # batch_per_class should be even
        self.subidx_shuffle = list()
        for cls_idx in range(self.nsclass):
            embed_tmp = self.embed[self.subidx[cls_idx]] # [batch_per_class, nembeddings]
            self.subidx_shuffle.append(self.subidx[cls_idx][np.concatenate(clustering_by_two(embed_tmp)[0], axis=0)])

        self.subidx_shuffle = np.concatenate(self.subidx_shuffle, axis=0)

        return self.image[self.subidx_shuffle], self.label[self.subidx_shuffle].astype('int32')

    def next_batch(self, batch_size, crop_size=None):
        '''
        Make batch data which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
            crop_size - list of  two elements
                defaults to be None for non cropping
        Return:
            image - batch_size image
            label - batch_size label
        '''
        assert batch_size%self.nsclass == 0, "Batchsize(%d) should be divided by nsclass(%d)"%(batch_size, self.nsclass)

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first
        
        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class

                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.subidx = np.concatenate(self.subidx, axis=0)

        if crop_size is None:
            return self.image[self.subidx], self.label[self.subidx].astype('int32')
        else:
            return np_random_crop_4d(self.image[self.subidx], crop_size), self.label[self.subidx].astype('int32')

    def next_batch_idx(self, batch_size):
        '''
        Make idx data for batch which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            image - batch_size image
            label - batch_size label
        '''
        assert batch_size%self.nsclass == 0, "Batchsize(%d) should be divided by nsclass(%d)"%(batch_size, self.nsclass)

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0: np.random.shuffle(self.fullidx[index]) # shuffle first
        
        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class

                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.subidx = np.concatenate(self.subidx, axis=0)
        return self.subidx

class NpairDatamanager(object):
    def __init__(self, image, label, nclass, nsclass):
        self.image = image
        self.label = label
        self.nclass = nclass
        self.nsclass = nsclass

        self.ndata = len(self.label)

        # list with [self.nclass] each element = idx set of which label is cls_idx
        # initialize
        self.label_idx_set = list()
        for cls_idx in range(self.nclass): self.label_idx_set.append(list())
        # append
        for d_idx in range(self.ndata): self.label_idx_set[self.label[d_idx]].append(d_idx)
        # to numpy
        for cls_idx in range(self.nclass): self.label_idx_set[cls_idx] = np.array(self.label_idx_set[cls_idx])

        self.valid_class_set = [cls_idx for cls_idx in range(self.nclass) if len(self.label_idx_set[cls_idx])>1]
        self.ndata_idx = np.array([len(vlist) for vlist in self.label_idx_set])
        self.fullidx = [np.arange(self.ndata_idx[index], dtype=np.int32) for index in range(self.nclass)]
        self.start = np.zeros(self.nclass, dtype=np.int32)
        self.end = np.zeros(self.nclass, dtype=np.int32)

    def print_shape(self):
        print("Image shape : {}".format(self.image.shape))
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata): counter[int(self.label)]+=1
        return counter

    def add_embed(self, embed):
        self.embed = embed

    def next_batch_with_embed(self, batch_size):
        '''
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            anc_img
            pos_img
            anc_label - label of anc img
            pos_label - label of pos img
                anc_label and pos_label is idential just for checking
        '''
        assert batch_size%(2*self.nsclass) == 0, "Batchsize(%d) should be multiple of (2*nsclass)(=%d)"%(batch_size, 2*self.nsclass) 

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first

        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class
                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        # self.subidx [batch_per_class] x nsclass
        # batch_per_class should be even
        self.subidx_shuffle = list()
        for idx in range(self.nsclass):
            embed_tmp = self.embed[self.subidx[idx]] # [batch_per_class, nembeddings]
            self.subidx_shuffle.append(self.subidx[idx][np.concatenate(clustering_by_two(embed_tmp)[0], axis=0)])

        self.anc_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==0] for v in self.subidx_shuffle], axis=0)
        self.pos_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==1] for v in self.subidx_shuffle], axis=0)

        assert len(self.anc_subidx)==len(self.pos_subidx), "Both anc and pos have same length"

        return self.image[self.anc_subidx],\
               self.image[self.pos_subidx],\
               self.label[self.anc_subidx].astype('int32'),\
               self.label[self.pos_subidx].astype('int32')

    def next_batch(self, batch_size, crop_size=None):
        '''
            crop
        Args:
            batch_size - int
                return batch size numbers of pairs
            crop_size - list of 2 elements
                defaults to be None for no clipping
        Return :
            anc_img
            pos_img
            anc_label - label of anc img
            pos_label - label of pos img
                anc_label and pos_label is idential just for checking
        '''
        assert batch_size%(2*self.nsclass) == 0, "Batchsize(%d) should be multiple of (2*nsclass)(=%d)"%(batch_size, 2*self.nsclass) 

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first

        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class
                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.anc_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==0] for v in self.subidx], axis=0)
        self.pos_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==1] for v in self.subidx], axis=0)

        assert len(self.anc_subidx)==len(self.pos_subidx), "Both anc and pos have same length"

        if crop_size is None:
            return self.image[self.anc_subidx],\
                   self.image[self.pos_subidx],\
                   self.label[self.anc_subidx].astype('int32'),\
                   self.label[self.pos_subidx].astype('int32')
        else:
            return np_random_crop_4d(self.image[self.anc_subidx], crop_size),\
                   np_random_crop_4d(self.image[self.pos_subidx], crop_size),\
                   self.label[self.anc_subidx].astype('int32'),\
                   self.label[self.pos_subidx].astype('int32')

    def next_batch_idx(self, batch_size):
        '''
            crop
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            anc_img
            pos_img
            anc_label - label of anc img
            pos_label - label of pos img
                anc_label and pos_label is idential just for checking
        '''
        assert batch_size%(2*self.nsclass) == 0, "Batchsize(%d) should be multiple of (2*nsclass)(=%d)"%(batch_size, 2*self.nsclass) 

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first

        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class
                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.anc_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==0] for v in self.subidx], axis=0)
        self.pos_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==1] for v in self.subidx], axis=0)

        return self.anc_subidx, self.pos_subidx

class TripletDatamanagerShort(object):
    def __init__(self, label, nclass, nsclass=2):
        '''
        Args:
            label - 
            nclass - number of total classes
            nsclass - When we select the batch, the number of classes it contain
        '''
        self.label = label
        self.nclass = nclass
        self.nsclass = nsclass 

        self.ndata = len(self.label)

        # list with [self.nclass] each element = idx set of which label is cls_idx
        # initialize
        
        self.label_idx_set = list()
        for cls_idx in range(self.nclass): self.label_idx_set.append(list())
        # append
        for d_idx in range(self.ndata): self.label_idx_set[self.label[d_idx]].append(d_idx)
        # to numpy
        for cls_idx in range(self.nclass): self.label_idx_set[cls_idx] = np.array(self.label_idx_set[cls_idx])
        
        self.valid_class_set = [cls_idx for cls_idx in range(self.nclass) if len(self.label_idx_set[cls_idx])>1]
        self.ndata_idx = np.array([len(v) for v in self.label_idx_set])
        self.fullidx = [np.arange(self.ndata_idx[cls_idx], dtype=np.int32) for cls_idx in range(self.nclass)]
        self.start = np.zeros(self.nclass, dtype=np.int32)
        self.end = np.zeros(self.nclass, dtype=np.int32)

    def print_shape(self):
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata): counter[int(self.label)]+=1
        return counter

    def change_nsclass(self, value):
        self.nsclass = value

    def next_batch_idx(self, batch_size):
        '''
        Make idx data for batch which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            label - batch_size label
        '''
        assert batch_size%self.nsclass == 0, "Batchsize(%d) should be divided by nsclass(%d)"%(batch_size, self.nsclass)

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0: np.random.shuffle(self.fullidx[index]) # shuffle first
        
        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class

                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.subidx = np.concatenate(self.subidx, axis=0)
        return self.subidx

class TripletDatamanagerDouble(object):
    def __init__(self, label1, label2, nclass1, nclass2, nsclass1, nsclass2, min_=2):
        self.label1 = label1
        self.label2 = label2
        self.nclass1 = nclass1
        self.nclass2 = nclass2
        self.nsclass1 = nsclass1
        self.nsclass2 = nsclass2

        assert len(self.label1)==len(self.label2), "label1 and label2 should have the same length"

        self.ndata = len(self.label1)
        # initialize
        self.label_idx_matrix = list()
        for cls_idx1 in range(self.nclass1): self.label_idx_matrix.append([list() for cls_idx2 in range(self.nclass2)])

        # append
        for d_idx in range(self.ndata): self.label_idx_matrix[self.label1[d_idx]][self.label2[d_idx]].append(d_idx)
        # to numpy
        for cls_idx1 in range(self.nclass1):
            for cls_idx2 in range(self.nclass2):
                self.label_idx_matrix[cls_idx1][cls_idx2] = np.array(self.label_idx_matrix[cls_idx1][cls_idx2])

        self.valid_class2_matrix = list()
        for cls_idx1 in range(self.nclass1):
            self.valid_class2_matrix.append([cls_idx2 for cls_idx2 in range(self.nclass2) if len(self.label_idx_matrix[cls_idx1][cls_idx2])>=min_])

        self.valid_class1_set = [cls_idx1 for cls_idx1 in range(self.nclass1) if len(self.valid_class2_matrix[cls_idx1])>=nsclass2]

        self.ndata_matrix = np.zeros([self.nclass1, self.nclass2], dtype=np.int32)
        for cls_idx1 in range(self.nclass1):
            for cls_idx2 in range(self.nclass2):
                self.ndata_matrix[cls_idx1][cls_idx2] = len(self.label_idx_matrix[cls_idx1][cls_idx2])

        self.fullidx = [[np.arange(self.ndata_matrix[cls_idx1][cls_idx2], dtype=np.int32) for cls_idx2 in range(self.nclass2)] for cls_idx1 in range(self.nclass1)]
        self.start = np.zeros([self.nclass1, self.nclass2], dtype=np.int32)
        self.end = np.zeros([self.nclass1, self.nclass2], dtype=np.int32)

    def print_shape(self):
        print("Label1 shape : {}".format(self.label1.shape))
        print("Label2 shape : {}".format(self.label2.shape))

    def next_batch_idx(self, batch_size):
        '''
        Make idx data for batch which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            batch_idx
        '''
        assert batch_size%(self.nsclass1*self.nsclass2) == 0, "Batchsize(%d) should be multiple of (nsclass1*nsclass2)(=%d)"%(batch_size, self.nsclass1*self.nsclass2) 

        batch_per_class = batch_size//(self.nsclass1*self.nsclass2)

        sclass1 = np.array(random.sample(self.valid_class1_set, self.nsclass1)) # nsclass1
        sclass2_set = [np.array(random.sample(self.valid_class2_matrix[cls_idx1], self.nsclass2)) for cls_idx1 in sclass1]
         
        self.subidx = list()
        for idx1 in range(self.nsclass1):
            cls_idx1 = sclass1[idx1]
            for idx2 in range(self.nsclass2):
                cls_idx2= sclass2_set[idx1][idx2]
                if self.start[cls_idx1][cls_idx2]==0 and self.end[cls_idx1][cls_idx2]==0:
                    np.random.shuffle(self.fullidx[cls_idx1][cls_idx2]) # shuffle first
                if self.end[cls_idx1][cls_idx2] + batch_per_class > self.ndata_matrix[cls_idx1][cls_idx2]:
                    self.start[cls_idx1][cls_idx2] = self.end[cls_idx1][cls_idx2]
                    self.end[cls_idx1][cls_idx2] = (self.end[cls_idx1][cls_idx2] + batch_per_class)%self.ndata_matrix[cls_idx1][cls_idx2]
                    self.subidx.append(self.label_idx_matrix[cls_idx1][cls_idx2][
                                            np.append(self.fullidx[cls_idx1][cls_idx2][self.start[cls_idx1][cls_idx2]:self.ndata_matrix[cls_idx1][cls_idx2]], self.fullidx[cls_idx1][cls_idx2][0:self.end[cls_idx1][cls_idx2]])
                                            ])
                    self.start[cls_idx1][cls_idx2] = 0
                    self.end[cls_idx1][cls_idx2] = 0
                else:
                    self.start[cls_idx1][cls_idx2] = self.end[cls_idx1][cls_idx2]
                    self.end[cls_idx1][cls_idx2] += batch_per_class
                    self.subidx.append(self.label_idx_matrix[cls_idx1][cls_idx2][self.fullidx[cls_idx1][cls_idx2][self.start[cls_idx1][cls_idx2]:self.end[cls_idx1][cls_idx2]]])

                    if self.end[cls_idx1][cls_idx2]==self.ndata_matrix[cls_idx1][cls_idx2]:
                        self.start[cls_idx1][cls_idx2] = 0
                        self.end[cls_idx1][cls_idx2] = 0

        self.subidx = np.concatenate(self.subidx, axis=0)
        return self.subidx

class NpairDatamanagerDouble(object):
    def __init__(self, label1, label2, nclass1, nclass2, nsclass1, nsclass2, min_=2):
        self.label1 = label1
        self.label2 = label2
        self.nclass1 = nclass1
        self.nclass2 = nclass2
        self.nsclass1 = nsclass1
        self.nsclass2 = nsclass2

        assert len(self.label1)==len(self.label2), "label1 and label2 should have the same length"

        self.ndata = len(self.label1)
        # initialize
        self.label_idx_matrix = list()
        for cls_idx1 in range(self.nclass1): self.label_idx_matrix.append([list() for cls_idx2 in range(self.nclass2)])

        # append
        for d_idx in range(self.ndata): self.label_idx_matrix[self.label1[d_idx]][self.label2[d_idx]].append(d_idx)
        # to numpy
        for cls_idx1 in range(self.nclass1):
            for cls_idx2 in range(self.nclass2):
                self.label_idx_matrix[cls_idx1][cls_idx2] = np.array(self.label_idx_matrix[cls_idx1][cls_idx2])

        self.valid_class2_matrix = list()
        for cls_idx1 in range(self.nclass1):
            self.valid_class2_matrix.append([cls_idx2 for cls_idx2 in range(self.nclass2) if len(self.label_idx_matrix[cls_idx1][cls_idx2])>=min_])

        self.valid_class1_set = [cls_idx1 for cls_idx1 in range(self.nclass1) if len(self.valid_class2_matrix[cls_idx1])>=nsclass2]

        self.ndata_matrix = np.zeros([self.nclass1, self.nclass2], dtype=np.int32)
        for cls_idx1 in range(self.nclass1):
            for cls_idx2 in range(self.nclass2):
                self.ndata_matrix[cls_idx1][cls_idx2] = len(self.label_idx_matrix[cls_idx1][cls_idx2])

        self.fullidx = [[np.arange(self.ndata_matrix[cls_idx1][cls_idx2], dtype=np.int32) for cls_idx2 in range(self.nclass2)] for cls_idx1 in range(self.nclass1)]
        self.start = np.zeros([self.nclass1, self.nclass2], dtype=np.int32)
        self.end = np.zeros([self.nclass1, self.nclass2], dtype=np.int32)

    def print_shape(self):
        print("Label1 shape : {}".format(self.label1.shape))
        print("Label2 shape : {}".format(self.label2.shape))

    def next_batch_idx(self, batch_size):
        '''
            crop
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            anc_idx
            pos_idx
        '''
        assert batch_size%(2*self.nsclass1*self.nsclass2) == 0, "Batchsize(%d) should be multiple of (2*nsclass1*nsclass2)(=%d)"%(batch_size, 2*self.nsclass1*self.nsclass2) 

        batch_per_class = batch_size//(self.nsclass1*self.nsclass2)

        sclass1 = np.array(random.sample(self.valid_class1_set, self.nsclass1)) # nsclass1
        sclass2_set = [np.array(random.sample(self.valid_class2_matrix[cls_idx1], self.nsclass2)) for cls_idx1 in sclass1]
         
        self.subidx = list()
        for idx1 in range(self.nsclass1):
            cls_idx1 = sclass1[idx1]
            for idx2 in range(self.nsclass2):
                cls_idx2= sclass2_set[idx1][idx2]
                if self.start[cls_idx1][cls_idx2] == 0 and self.end[cls_idx1][cls_idx2] ==0:
                    np.random.shuffle(self.fullidx[cls_idx1][cls_idx2]) # shuffle first
                if self.end[cls_idx1][cls_idx2] + batch_per_class > self.ndata_matrix[cls_idx1][cls_idx2]:
                    self.start[cls_idx1][cls_idx2] = self.end[cls_idx1][cls_idx2]
                    self.end[cls_idx1][cls_idx2] = (self.end[cls_idx1][cls_idx2] + batch_per_class)%self.ndata_matrix[cls_idx1][cls_idx2]

                    self.subidx.append(self.label_idx_matrix[cls_idx1][cls_idx2][
                                            np.append(self.fullidx[cls_idx1][cls_idx2][self.start[cls_idx1][cls_idx2]:self.ndata_matrix[cls_idx1][cls_idx2]], self.fullidx[cls_idx1][cls_idx2][0:self.end[cls_idx1][cls_idx2]])
                                            ])
                    self.start[cls_idx1][cls_idx2] = 0
                    self.end[cls_idx1][cls_idx2] = 0
                else:
                    self.start[cls_idx1][cls_idx2] = self.end[cls_idx1][cls_idx2]
                    self.end[cls_idx1][cls_idx2] += batch_per_class
                    self.subidx.append(self.label_idx_matrix[cls_idx1][cls_idx2][self.fullidx[cls_idx1][cls_idx2][self.start[cls_idx1][cls_idx2]:self.end[cls_idx1][cls_idx2]]])

                    if self.end[cls_idx1][cls_idx2]==self.ndata_matrix[cls_idx1][cls_idx2]:
                        self.start[cls_idx1][cls_idx2] = 0
                        self.end[cls_idx1][cls_idx2] = 0

        self.anc_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==0] for v in self.subidx], axis=0)
        self.pos_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==1] for v in self.subidx], axis=0)
        return self.anc_subidx, self.pos_subidx

class NpairDatamanagerShort(object):
    def __init__(self, label, nclass, nsclass):
        self.label = label
        self.nclass = nclass
        self.nsclass = nsclass

        self.ndata = len(self.label)

        # list with [self.nclass] each element = idx set of which label is cls_idx
        # initialize
        self.label_idx_set = list()
        for cls_idx in range(self.nclass): self.label_idx_set.append(list())
        # append
        for d_idx in range(self.ndata): self.label_idx_set[self.label[d_idx]].append(d_idx)
        # to numpy
        for cls_idx in range(self.nclass): self.label_idx_set[cls_idx] = np.array(self.label_idx_set[cls_idx])

        self.valid_class_set = [cls_idx for cls_idx in range(self.nclass) if len(self.label_idx_set[cls_idx])>1]
        self.ndata_idx = np.array([len(vlist) for vlist in self.label_idx_set])
        self.fullidx = [np.arange(self.ndata_idx[index], dtype=np.int32) for index in range(self.nclass)]
        self.start = np.zeros(self.nclass, dtype=np.int32)
        self.end = np.zeros(self.nclass, dtype=np.int32)

    def print_shape(self):
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata): counter[int(self.label)]+=1
        return counter

    def next_batch_idx(self, batch_size):
        '''
            crop
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            anc_img
            pos_img
            anc_label - label of anc img
            pos_label - label of pos img
                anc_label and pos_label is idential just for checking
        '''
        assert batch_size%(2*self.nsclass) == 0, "Batchsize(%d) should be multiple of (2*nsclass)(=%d)"%(batch_size, 2*self.nsclass) 

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first

        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class
                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.anc_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==0] for v in self.subidx], axis=0)
        self.pos_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==1] for v in self.subidx], axis=0)
        return self.anc_subidx, self.pos_subidx

DATAMANAGER_DICT = {
    'basic' : BasicDatamanager,
    'contrast' : ContrastDatamanager,
    'triplet' : TripletDatamanager,
    'npair' : NpairDatamanager,
    'triplet_s' : TripletDatamanagerShort,
    'npair_s' : NpairDatamanagerShort,
    'triplet_d' : TripletDatamanagerDouble,
    'npair_d' : NpairDatamanagerDouble
    }

if __name__=='__main__':
    import sys
    sys.path.append('../configs')
    sys.path.append('../utils')
    sys.path.append('../tfops')

    # utils
    from reader import read_npy
    # config
    from path import CIFARPROCESSED
    from info import CIFARNCLASS

    val_embed = read_npy(CIFARPROCESSED+'val_image.npy')    
    val_label = read_npy(CIFARPROCESSED+'val_label.npy')    

    cifar = TripletDatamanager(val_embed, val_label, CIFARNCLASS, nsclass=10) 
    count = np.zeros(cifar.nclass)
    nbatch = cifar.ndata//50+1
    for i in range(nbatch):
        _, label = cifar.next_batch(50)
        for index in range(len(label)):
            count[label[index]]+=1
    print(count)

    cifar = NpairDatamanager(val_embed, val_label, CIFARNCLASS, nsclass=4)

    _, _, anc_l, pos_l = cifar.next_batch(32)
    print(anc_l)
    print(pos_l)
