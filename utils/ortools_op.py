import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.np_op import plabel2subset

from ortools.graph import pywrapgraph

import numpy as np
import copy

def greedy_max_matching_label_solver_hungarian_initialization(array, plamb, labels, k, value=10000):
    '''
    NIPS2018/idea2.pdf/Method1

    Args:
        array - 2D Numpy array [nworkers, ntasks]
        plamb - float
        labels - 1D Numpy array [nworkers]
        k - int
        value - int
    Return:
        objective - 2D Numpy array [nworkers, ntasks]
            binary array 
        value - float value of energy
    '''
    nworkers, ntasks = array.shape

    subsets = plabel2subset(labels=labels)

    nsubsets = len(subsets)

    array_p = value*array
    array_p = -array_p # potential to cost
    array_p = array_p.astype(np.int32)

    occupy = -1*np.ones([nworkers], dtype=np.int32)
    occupy = np.array(occupy)
    occupy = (occupy.tolist())

    for s_idx in range(nsubsets):
        for w_idx in subsets[s_idx]:occupy[w_idx] = s_idx

    for w_idx in range(nworkers): assert occupy[w_idx]!=-1, "There is an only label element"

    source = 0
    sink = 1 + nworkers + nsubsets*ntasks
    pcost = int(plamb*value)

    supplies = [nworkers*k]+(sink-1)*[0]+[-nworkers*k]

    start_nodes = list()
    end_nodes = list()
    capacities = list()
    costs = list()

    # source2workers
    for w_idx in range(nworkers):
        start_nodes.append(source)
        end_nodes.append(w_idx+1)
        capacities.append(k)
        costs.append(0)

    # workers2subsets
    for w_idx in range(nworkers):
        s_idx = occupy[w_idx]
        for t_idx in range(ntasks):
            start_nodes.append(1 + w_idx) # workers
            end_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
            capacities.append(1)
            costs.append(array_p[w_idx][t_idx])

    # subsets2sink
    for s_idx in range(nsubsets):
        for t_idx in range(ntasks):
            start_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
            end_nodes.append(sink)
            capacities.append(1)
            costs.append(0)

    nedge = len(start_nodes)
    nvertex = len(supplies)

    costs = np.array(costs)
    costs = (costs.tolist())

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    for idx in range(nedge):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[idx], end_nodes[idx], capacities[idx], costs[idx])
    for idx in range(nvertex):
        min_cost_flow.SetNodeSupply(idx, supplies[idx])

    min_cost_flow.Solve()
    results = list()

    for arc in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:
            if min_cost_flow.Flow(arc)>0:
                start = min_cost_flow.Tail(arc)-1
                if start<nworkers:
                    end = (min_cost_flow.Head(arc)-nworkers-1)%ntasks
                    results.append([start, end])

    objective = dict()
    usage_count = np.zeros([ntasks])
    label_usage_count = dict()
    for w_idx, t_idx in results:
        if w_idx not in objective.keys() : objective[w_idx] = list() 
        objective[w_idx].append(t_idx)
        usage_count[t_idx]+=1
        w_label = labels[w_idx]  
        if w_label not in label_usage_count.keys(): label_usage_count[w_label] = np.zeros([ntasks])
        label_usage_count[w_label][t_idx]+=1
        
    for w_idx in range(nworkers):
        w_label = labels[w_idx]
        tmp = -array[w_idx]+plamb*(usage_count-label_usage_count[w_label])
        t_new_idx_set = np.argsort(tmp)[:k] # greedy choice
        t_old_idx_set = objective[w_idx]

        for t_old_idx in t_old_idx_set:
            label_usage_count[w_label][t_old_idx]-=1
            usage_count[t_old_idx] -= 1

        objective[w_idx] = list()
        for t_new_idx in t_new_idx_set:
            label_usage_count[w_label][t_new_idx]+=1
            usage_count[t_new_idx] += 1
            objective[w_idx].append(t_new_idx)

    new_objective = np.zeros([nworkers, ntasks])

    for w_idx in range(nworkers):
        for t_idx in objective[w_idx]:
            new_objective[w_idx][t_idx]=1

    return new_objective

def solve_maxmatching_strict_intraclass(array, labels, plamb, value=10000):
    '''
    NIPS2018/idea2/Method2
    same class select different bucket indefinitely

    Args:
        array - Numpy 2D array [nworkers, ntasks]
        labels - Numpy 1D array [nworkers]
            which have same labels have strictly different task
        plamb - float
        value - int
    '''
    nworkers, ntasks = array.shape

    subsets = plabel2subset(labels=labels)

    nsubsets = len(subsets)

    array_p = value*array
    array_p = -array_p # potential to cost
    array_p = array_p.astype(np.int32)

    occupy = -1*np.ones([nworkers], dtype=np.int32)
    occupy = np.array(occupy)
    occupy = (occupy.tolist())

    for s_idx in range(nsubsets):
        for w_idx in subsets[s_idx]:occupy[w_idx] = s_idx

    for w_idx in range(nworkers): assert occupy[w_idx]!=-1, "There is an only label element"

    source = 0
    sink = 1 + nworkers + nsubsets*ntasks + ntasks
    pcost = int(plamb*value)

    supplies = [nworkers]+(sink-1)*[0]+[-nworkers]

    start_nodes = list()
    end_nodes = list()
    capacities = list()
    costs = list()

    # source2workers
    for w_idx in range(nworkers):
        start_nodes.append(source)
        end_nodes.append(w_idx+1)
        capacities.append(1)
        costs.append(0)

    # workers2subsets
    for w_idx in range(nworkers):
        s_idx = occupy[w_idx]
        for t_idx in range(ntasks):
            start_nodes.append(1 + w_idx) # workers
            end_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
            capacities.append(1)
            costs.append(array_p[w_idx][t_idx])

    # subsets2tasks
    for s_idx in range(nsubsets):
        for t_idx in range(ntasks):
            start_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
            end_nodes.append(1 + nworkers + nsubsets*ntasks + t_idx) # tasks
            capacities.append(1)
            costs.append(0)

    # tasks2sink
    for t_idx in range(ntasks):
        for w_idx in range(nworkers):
            start_nodes.append(1 + nworkers + ntasks*nsubsets + t_idx)
            end_nodes.append(sink)
            capacities.append(1)
            costs.append(w_idx*pcost)

    nedge = len(start_nodes)
    nvertex = len(supplies)

    costs = np.array(costs)
    costs = (costs.tolist())

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    for idx in range(nedge):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[idx], end_nodes[idx], capacities[idx], costs[idx])
    for idx in range(nvertex):
        min_cost_flow.SetNodeSupply(idx, supplies[idx])

    min_cost_flow.Solve()
    results = list()

    for arc in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:
            if min_cost_flow.Flow(arc)>0:
                start = min_cost_flow.Tail(arc)-1
                if start<nworkers:
                    end = (min_cost_flow.Head(arc)-nworkers-1)%ntasks
                    results.append([start, end])
    return results

def solve_maxmatching_soft_intraclass_multiselect_greedy(array, k, labels, plamb1, plamb2, value=10000):
    '''
    NIPS2018/idea1.pdf/objective2, NIPS2018/idea2.pdf/Method3
    Args:
        array - Numpy 2D array [nworkers, ntasks]
        k - int
        labels - Numpy 1D array [nworkers]
            Which have same label will have additional pairwise cost
        plamb1 - float
            subsets to tasks
        plamb2 - float
            tasks to sink
        value - int
    '''
    nworkers, ntasks = array.shape
    objective = np.zeros([nworkers, ntasks])
    array_copy = copy.copy(array)

    subsets = plabel2subset(labels=labels)
    nsubsets = len(subsets)

    occupy = -1*np.ones([nworkers], dtype=np.int32)
    occupy = np.array(occupy)
    occupy = (occupy.tolist())

    nsubset_list = list()

    for s_idx in range(nsubsets):
        nsubset_list.append(len(subsets[s_idx]))
        for w_idx in subsets[s_idx]: occupy[w_idx] = s_idx

    usage1 = np.zeros([nsubsets, ntasks])
    usage2 = np.zeros([ntasks])

    for _ in range(k):
        for w_idx in range(nworkers):
            o_idx = occupy[w_idx]
            tmp = -array_copy[w_idx]+plamb2*(usage2-objective[w_idx])
            if o_idx!=-1:
                tmp += plamb1*(usage1[o_idx]-objective[w_idx])
            t_idx = np.argmin(tmp)
            objective[w_idx][t_idx]=1
            usage2[t_idx] = usage2[t_idx]+1
            if o_idx!=-1:
                usage1[o_idx][t_idx] = usage1[o_idx][t_idx]+1
            array_copy[w_idx][t_idx]=float('-inf')
    return objective

def solve_maxmatching_soft_intraclass_multiselect(array, k, labels, plamb1, plamb2, value=10000):
    '''
    NIPS2018/idea1.pdf/objective2, NIPS2018/idea2.pdf/Method3
    Args:
        array - Numpy 2D array [nworkers, ntasks]
        k - int
        labels - Numpy 1D array [nworkers]
            Which have same label will have additional pairwise cost
        plamb1 - float
            subsets to tasks
        plamb2 - float
            tasks to sink
        value - int
    '''
    nworkers, ntasks = array.shape
    subsets = plabel2subset(labels=labels)
    nsubsets = len(subsets)

    array_p = value*array
    array_p = -array_p # potential to cost
    array_p = array_p.astype(np.int32)

    occupy = -1*np.ones([nworkers], dtype=np.int32)
    occupy = np.array(occupy)
    occupy = (occupy.tolist())

    nsubset_list = list()

    for s_idx in range(nsubsets):
        nsubset_list.append(len(subsets[s_idx]))
        for w_idx in subsets[s_idx]: occupy[w_idx] = s_idx

    source = 0
    sink = 1 + nworkers + nsubsets*ntasks + ntasks
    pcost1 = int(plamb1*value)
    pcost2 = int(plamb2*value)

    supplies = [nworkers*k]+(sink-1)*[0]+[-nworkers*k] # changed here

    start_nodes = list()
    end_nodes = list()
    capacities = list()
    costs = list()

    # source2workers
    for w_idx in range(nworkers):
        start_nodes.append(source)
        end_nodes.append(w_idx+1)
        capacities.append(k) # changed here
        costs.append(0)

    # workers2subsets or tasks
    for w_idx in range(nworkers):
        if occupy[w_idx]==-1: # to tasks
            for t_idx in range(ntasks):
                start_nodes.append(1 + w_idx) # workers
                end_nodes.append(1 + nworkers + nsubsets*ntasks + t_idx) # tasks
                capacities.append(1)
                costs.append(array_p[w_idx][t_idx])
        else:
            s_idx = occupy[w_idx]
            for t_idx in range(ntasks):
                start_nodes.append(1 + w_idx) # workers
                end_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
                capacities.append(1)
                costs.append(array_p[w_idx][t_idx])

    # subsets2tasks
    for s_idx in range(nsubsets):
        nworkers_subset = nsubset_list[s_idx]
        for t_idx in range(ntasks):
            for w_idx in range(nworkers_subset):
                start_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
                end_nodes.append(1 + nworkers + nsubsets*ntasks + t_idx) # tasks
                capacities.append(1)
                costs.append(w_idx*pcost1)

    # tasks2sink
    for t_idx in range(ntasks):
        for w_idx in range(nworkers):
            start_nodes.append(1 + nworkers + ntasks*nsubsets + t_idx)
            end_nodes.append(sink)
            capacities.append(1)
            costs.append(w_idx*pcost2)

    nedge = len(start_nodes)
    nvertex = len(supplies)

    costs = np.array(costs)
    costs = (costs.tolist())

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    for idx in range(nedge):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[idx], end_nodes[idx], capacities[idx], costs[idx])
    for idx in range(nvertex):
        min_cost_flow.SetNodeSupply(idx, supplies[idx])

    min_cost_flow.Solve()
    results = list()

    for arc in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:
            if min_cost_flow.Flow(arc)>0:
                start = min_cost_flow.Tail(arc)-1
                if start<nworkers:
                    end = (min_cost_flow.Head(arc)-nworkers-1)%ntasks
                    results.append([start, end])
    return results

def solve_maxmatching_soft_intraclass(array, labels, plamb1, plamb2, value=10000):
    '''
    NIPS2018/idea1.pdf/objective2, NIPS2018/idea2.pdf/Method3
    Args:
        array - Numpy 2D array [nworkers, ntasks]
        labels - Numpy 1D array [nworkers]
            Which have same label will have additional pairwise cost
        plamb1 - float
            subsets to tasks
        plamb2 - float
            tasks to sink
        value - int
    '''
    return solve_maxmatching_soft_intraclass_multiselect(array=array, k=1, labels=labels, plamb1=plamb1, plamb2=plamb2, value=value)

def solvemaxmatching_label(up, plabel, max_label, plamb=0.0): 
    '''SolveMaxMatching wrapper
    Args:
        up - Numpy 2D array [nworkers, ntasks]
            unary potential
        plabel - Numpy 1D array [nworkers]
            label to be maximized
        max_label - int
        plamb - float
            defaults to be 0.0
    ''' 
    nworkers, ntasks = up.shape
    assert nworkers==len(plabel), "Wrong plabel length"
    
    objective = np.zeros([nworkers, ntasks]) 

    label2idx_set = list()
    for _ in range(max_label): label2idx_set.append(list())

    for w_idx in range(nworkers):
        label2idx_set[plabel[w_idx]].append(w_idx)

    for l_idx in range(max_label):
        idx_set = label2idx_set[l_idx]
        nsworkers = len(idx_set)

        mcf  = SolveMaxMatching(nworkers=nsworkers, ntasks=ntasks, k=1, pairwise_lamb=plamb)   
        for i, j in mcf.solve(up[idx_set]):
            objective[idx_set[i]][j]=1
    return objective

class SolveMaxMatching:
    def __init__(self, nworkers, ntasks, k, value=10000, pairwise_lamb=0.1):
        '''
        This can be used when nworkers*k > ntasks
        Args:
            nworkers - int
            ntasks - int
            k - int
            value - int 
                should be large defaults to be 10000

            pairwise_lamb - int

        '''
        self.nworkers = nworkers
        self.ntasks = ntasks
        self.value = value
        self.k = k

        self.source = 0
        self.sink = self.nworkers+self.ntasks+1

        self.pairwise_cost = int(pairwise_lamb*value)

        self.supplies = [self.nworkers*self.k]+(self.ntasks+self.nworkers)*[0]+[-self.nworkers*self.k]
        self.start_nodes = list()
        self.end_nodes = list() 
        self.capacities = list()
        self.common_costs = list()

        for work_idx in range(self.nworkers):
            self.start_nodes.append(self.source)
            self.end_nodes.append(work_idx+1)
            self.capacities.append(self.k)
            self.common_costs.append(0)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(self.nworkers+1+task_idx)
                self.end_nodes.append(self.sink)
                self.capacities.append(1)
                self.common_costs.append(work_idx*self.pairwise_cost)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(work_idx+1)
                self.end_nodes.append(self.nworkers+1+task_idx)
                self.capacities.append(1)

        self.nnodes = len(self.start_nodes)

    def solve(self, array):
        '''
        Args:
            array - Numpy 2D array [nworkers, ntasks]
        Return:
            results
        '''
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)

        array_p = self.value*array
        array_p = -array_p # potential to cost
        array_p = array_p.astype(np.int32)

        costs = copy.copy(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs.append(array_p[work_idx][task_idx])

        costs = np.array(costs)
        costs = (costs.tolist())

        assert len(costs)==self.nnodes, "Length of costs should be {} but {}".format(self.nnodes, len(costs))

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for idx in range(self.nnodes):
             min_cost_flow.AddArcWithCapacityAndUnitCost(self.start_nodes[idx], self.end_nodes[idx], self.capacities[idx], costs[idx])
        for idx in range(self.ntasks+self.nworkers+2):
            min_cost_flow.SetNodeSupply(idx, self.supplies[idx])

        min_cost_flow.Solve()
        results = list()
        for arc in range(min_cost_flow.NumArcs()):
            if min_cost_flow.Tail(arc)!=self.source and min_cost_flow.Head(arc)!=self.sink:
                if min_cost_flow.Flow(arc)>0:
                    results.append([min_cost_flow.Tail(arc)-1, min_cost_flow.Head(arc)-self.nworkers-1])
        return results

    def solve_w_label(self, array, plabel, pidx):
        '''
        Args:
            array - Numpy 2D array [nworkers, ntasks]
            plabel - Numpy 1D array [nworkers]
            pidx - int
        Return:
            results
        '''
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)
        assert len(plabel) == self.nworkers, "Wrong plabel shape, it should be length(={})".format(self.nworkers)

        nsworkers = 0

        capacities = copy.copy(self.capacities)
        for work_idx in range(self.nworkers):
            if plabel[work_idx]==pidx:
                nsworkers+=1
                capacities[work_idx] = self.k
            else:
                capacities[work_idx] = 0

        supplies = [nsworkers*self.k]+(self.ntasks+self.nworkers)*[0]+[-nsworkers*self.k]

        array_p = self.value*array
        array_p = -array_p # potential to cost
        array_p = array_p.astype(np.int32)

        costs = copy.copy(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs.append(array_p[work_idx][task_idx])

        costs = np.array(costs)
        costs = (costs.tolist())

        assert len(costs)==self.nnodes, "Length of costs should be {} but {}".format(self.nnodes, len(costs))

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for idx in range(self.nnodes):
             min_cost_flow.AddArcWithCapacityAndUnitCost(self.start_nodes[idx], self.end_nodes[idx], capacities[idx], costs[idx])
        for idx in range(self.ntasks+self.nworkers+2):
            min_cost_flow.SetNodeSupply(idx, supplies[idx])

        min_cost_flow.Solve()
        results = list()
        for arc in range(min_cost_flow.NumArcs()):
            if min_cost_flow.Tail(arc)!=self.source and min_cost_flow.Head(arc)!=self.sink:
                if min_cost_flow.Flow(arc)>0:
                    results.append([min_cost_flow.Tail(arc)-1, min_cost_flow.Head(arc)-self.nworkers-1])
        return results

class SolveMaxMatchingHungarian:
    def __init__(self, nworkers, ntasks, k, value=10000):
        '''
        This class should be k*nworkers < ntasks
        Args:
            nworkers - int
            ntasks - int 
            k - int
                number of activations
        '''
        self.nworkers = nworkers
        self.ntasks = ntasks
        self.value = value
        self.k = k
        
        self.source = 0
        self.sink = self.nworkers+self.ntasks+1

        self.supplies = [self.nworkers*self.k]+(self.ntasks+self.nworkers)*[0]+[-self.nworkers*k]
        self.start_nodes = list()
        self.end_nodes = list() 
        self.capacities = list()
        self.common_costs = list()

        for work_idx in range(self.nworkers):
            self.start_nodes.append(self.source)
            self.end_nodes.append(work_idx+1)
            self.capacities.append(self.k)
            self.common_costs.append(0)

        for task_idx in range(self.ntasks):
            self.start_nodes.append(self.nworkers+1+task_idx)
            self.end_nodes.append(self.sink)
            self.capacities.append(1)
            self.common_costs.append(0)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(work_idx+1)
                self.end_nodes.append(self.nworkers+1+task_idx)
                self.capacities.append(1)


        self.nnodes = len(self.start_nodes)

    def solve(self, array):
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)

        self.array = self.value*array
        self.array = -self.array
        self.array = self.array.astype(np.int32)

        costs = copy.copy(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs.append(self.array[work_idx][task_idx])

        costs = np.array(costs)
        costs = (costs.tolist())

        assert len(costs)==self.nnodes, "Length of costs should be {} but {}".format(self.nnodes, len(costs))

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for idx in range(self.nnodes):
             min_cost_flow.AddArcWithCapacityAndUnitCost(self.start_nodes[idx], self.end_nodes[idx], self.capacities[idx], costs[idx])
        for idx in range(self.ntasks+self.nworkers+2):
            min_cost_flow.SetNodeSupply(idx, self.supplies[idx])

        min_cost_flow.Solve()
        results = list()
        for arc in range(min_cost_flow.NumArcs()):
            if min_cost_flow.Tail(arc)!=self.source and min_cost_flow.Head(arc)!=self.sink:
                if min_cost_flow.Flow(arc)>0:
                    results.append([min_cost_flow.Tail(arc)-1, min_cost_flow.Head(arc)-self.nworkers-1])

        return results

