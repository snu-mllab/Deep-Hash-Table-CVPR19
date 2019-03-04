from ortools_op import solve_maxmatching_soft_intraclass, solve_maxmatching_soft_intraclass_multiselect, solve_maxmatching_strict_intraclass,\
                       solve_maxmatching_soft_intraclass_multiselect_greedy, solvemaxmatching_label, SolveMaxMatching,\
                       greedy_max_matching_label_solver_hungarian_initialization,\
                       SolveMaxMatchingHungarian
from pygco_op import DiscreteEnergyMinimize
from ortools.graph import pywrapgraph
from np_op import greedy_max_matching_solver,\
                  greedy_max_matching_label_solver, greedy_max_matching_label_solver_iter,\
                  greedy_max_matching_label_solver_k, greedy_max_matching_label_solver_k_iter,\
                  plabel2subset
import numpy as np
import time

def results2objective(results, nworkers, ntasks):
    objective = np.zeros([nworkers, ntasks]) 
    for i,j in results: objective[i][j]=1
    return objective

def get_value(potential, objective, plamb):
    nworkers, ntasks = potential.shape
    usage_count = np.zeros(ntasks)
    value = 0
    for i in range(nworkers):
        for j in range(ntasks):
            if objective[i][j]==1:
                usage_count[j]+=1
                value += -potential[i][j]
    for i in range(ntasks): value += plamb*usage_count[i]*(usage_count[i]-1)/2
    return value

def get_value_label(potential, objective, plamb, labels):
    nworkers, ntasks = potential.shape
    label_usage_count = dict()
    for w_idx in range(nworkers):
        w_label = labels[w_idx]
        if w_label in label_usage_count.keys():continue
        label_usage_count[w_label] = np.zeros([ntasks])
    usage_count = np.zeros(ntasks)
    value = 0
    for w_idx in range(nworkers):
        for t_idx in range(ntasks):
            if objective[w_idx][t_idx]==1:
                w_label = labels[w_idx]
                value += -potential[w_idx][t_idx]
                value += plamb*(usage_count[t_idx]-label_usage_count[w_label][t_idx])
                usage_count[t_idx]+=1
                label_usage_count[w_label][t_idx]+=1
    return value

def test1():
    '''
    test for SolveMaxMatching
    '''
    pairwise_lamb = 0.0
    nworkers = 5
    ntasks = 6
    k = 2

    unary_potential = np.random.random([nworkers, ntasks])
    acf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=pairwise_lamb)   
    mcf_results = mcf.solve(unary_potential)
    objective = results2objective(results=mcf_results, nworkers=nworkers, ntasks=ntasks)

    print("unary potential:\n{}\nobjective:\n{}".format(unary_potential, objective))

def test2():
    '''
    test for SolveMaxMatching, solve_w_label
    '''
    plamb = 0.0
    nworkers = 5
    ntasks = 4
    max_label = 2

    unary_potential = np.random.random([nworkers, ntasks])
    plabel = np.random.randint(max_label, size=nworkers)

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=1, pairwise_lamb=plamb) 
    print("unary potential :\n{}\nplabel:\n{}".format(unary_potential, plabel))
    for p_idx in range(max_label):
        objective = results2objective(results=mcf.solve_w_label(unary_potential, plabel, p_idx), nworkers=nworkers,ntasks=ntasks)
        print("p_idx : {}\nobjective\n{}".format(p_idx, objective))

def test3():
    '''Speed check
    test for SolveMaxMatching, solve_w_label
    '''
    # hyper parameter setting
    plamb = 0.0
    nworkers = 64
    ntasks = 8
    max_label = 8
    max_iter = 20

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=1, pairwise_lamb=plamb)   
    
    w_label_time = 0
    wo_label_time = 0
    for _ in range(max_iter):
        unary_potential = np.random.random([nworkers, ntasks])
        plabel = np.random.randint(max_label, size=nworkers)

        start_time = time.time()
        for p_idx in range(max_label):
            mcf.solve_w_label(unary_potential, plabel, p_idx)
        end_time = time.time()
        w_label_time += end_time-start_time

        start_time = time.time()
        mcf.solve(unary_potential)
        end_time = time.time()
        wo_label_time += end_time-start_time
    print("w label time : {} sec\nwo label time : {} sec".format(w_label_time, wo_label_time))

def test4():
    '''Speed check
    solvematching_label
    solve_maxmatching_soft_intraclass
    '''
    plamb = 0.0
    nworkers = 64
    ntasks = 8
    max_label = 8
    max_iter = 20

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=1, pairwise_lamb=plamb)   
    
    w_label_time_method1 = 0
    w_label_time_method2 = 0
    wo_label_time = 0
    soft_intraclass_time = 0

    for _ in range(max_iter):
        unary_potential = np.random.random([nworkers, ntasks])
        plabel = np.random.randint(max_label, size=nworkers)

        start_time = time.time()
        for p_idx in range(max_label):
            results2objective(results=mcf.solve_w_label(unary_potential, plabel, p_idx), nworkers=nworkers,ntasks=ntasks)
        end_time = time.time()
        w_label_time_method1 += end_time-start_time

        start_time = time.time()
        solvemaxmatching_label(unary_potential, plabel, max_label, plamb=plamb)
        end_time = time.time()
        w_label_time_method2 += end_time-start_time

        start_time = time.time()
        results2objective(results=mcf.solve(unary_potential), nworkers=nworkers, ntasks=ntasks)
        end_time = time.time()
        wo_label_time += end_time-start_time

        start_time = time.time()
        results2objective(results=solve_maxmatching_soft_intraclass(unary_potential, plabel, plamb1=plamb, plamb2=0.0), nworkers=nworkers, ntasks=ntasks)
        end_time = time.time()
        soft_intraclass_time += end_time-start_time
    print("w label time method1 : {} sec\nw label time method2 : {} sec\nwo label time : {} sec\nsoft intraclass time : {} sec".format(w_label_time_method1, w_label_time_method2, wo_label_time, soft_intraclass_time))

def test5():
    '''Performance check'''
    # hyper parameter setting
    plamb = 0.1
    nworkers = 7
    ntasks = 5
    max_label = 2

    unary_potential = np.random.random([nworkers, ntasks])
    plabel = np.random.randint(max_label, size=nworkers)

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=1, pairwise_lamb=plamb)   
    objective = np.zeros([nworkers, ntasks], dtype=np.float32)
    for p_idx in range(max_label):
        results = mcf.solve_w_label(unary_potential, plabel, p_idx)
        for i, j in results:
            objective[i][j]=1
    print(objective)
    print(solvemaxmatching_label(unary_potential, plabel, max_label, plamb=plamb))
    print(results2objective(results=solve_maxmatching_soft_intraclass(unary_potential, plabel, plamb1=plamb, plamb2=0.0), nworkers=nworkers, ntasks=ntasks))

def test6():
    '''Speed check'''
    # hyper parameter setting
    plamb = 0.0
    nworkers = 64
    ntasks = 8
    max_label = 8
    max_iter = 20

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks*max_label, k=1, pairwise_lamb=plamb)   
    
    w_label_time = 0
    wo_label_time = 0

    for _ in range(max_iter):
        unary_potential = np.random.random([nworkers, ntasks])
        plabel = np.random.randint(max_label, size=nworkers)

        start_time = time.time()
        solvemaxmatching_label(unary_potential, plabel, max_label, plamb=plamb)
        end_time = time.time()
        w_label_time += end_time-start_time

        unary_potential = np.random.random([nworkers, ntasks*max_label])
        start_time = time.time()
        results2objective(results=mcf.solve(unary_potential), nworkers=nworkers, ntasks=ntasks*max_label)
        end_time = time.time()
        wo_label_time += end_time-start_time
    print("w label time : {}sec\nwo label time : {} sec".format(w_label_time, wo_label_time))

def test7():
    ''' test for solve_maxmatching_soft_intraclass
    '''
    nworkers = 10
    ntasks = 4

    unary_potential = np.random.random([nworkers, ntasks])

    plabels = np.zeros([nworkers])
    print("plabels : \n{}".format(plabels))

    results = solve_maxmatching_soft_intraclass(array=unary_potential, labels=plabels, plamb1=1.0, plamb2=1.0)
    objective = np.zeros([nworkers, ntasks])
    for i,j in results: objective[i][j]=1
    print("objective : \n{}".format(objective))

    for i in range(nworkers): plabels[i] = plabels[i]*ntasks+np.argmax(objective[i])

    print("plabels : \n{}".format(plabels))
    results = solve_maxmatching_soft_intraclass(array=unary_potential, labels=plabels, plamb1=1.0, plamb2=1.0)
    objective = np.zeros([nworkers, ntasks])
    for i,j in results: objective[i][j]=1
    print("objective : \n{}".format(objective))

    for i in range(nworkers): plabels[i] = plabels[i]*ntasks+np.argmax(objective[i])

    print("plabels : \n{}".format(plabels))
    results = solve_maxmatching_soft_intraclass(array=unary_potential, labels=plabels, plamb1=1.0, plamb2=1.0)
    objective = np.zeros([nworkers, ntasks])
    for i,j in results: objective[i][j]=1
    print("objective : \n{}".format(objective))

def test8():
    '''
    exp for rebutall
    '''
    pairwise_lamb = 1.0
    nworkers = 64
    ntasks = 64
    k = 1

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=pairwise_lamb)   
    dem = DiscreteEnergyMinimize(ntasks, pairwise_lamb)
    unary_potential = np.random.random([nworkers, ntasks])

    print("unary potential : {}".format(unary_potential))

    ab_start_time = time.time() 
    ab_objective = dem.solve(-unary_potential, np.ones([nworkers, nworkers]), k)
    ab_end_time = time.time() 

    mcf_start_time = time.time() 
    mcf_objective = results2objective(results=mcf.solve(unary_potential), nworkers=nworkers,ntasks=ntasks)
    mcf_end_time = time.time()
    greedy_start_time = time.time() 
    greedy_objective = greedy_max_matching_solver(array=unary_potential, plamb=pairwise_lamb, k=k)
    greedy_end_time = time.time() 

    unary_results = list()
    for i in range(nworkers):
        unary_sort = np.argsort(-unary_potential[i])[:k]
        for j in unary_sort:
            unary_results.append([i,j]) 
    unary_objective = results2objective(results=unary_results, nworkers=nworkers, ntasks=ntasks)
    ab_time = ab_end_time - ab_start_time
    mcf_time = mcf_end_time - mcf_start_time
    greedy_time = greedy_end_time - greedy_start_time
    print("mcf({}sec)\nobjective\n{}\nvalue : {}".format(mcf_time, mcf_objective, get_value(unary_potential, mcf_objective, pairwise_lamb)))
    print("ab({}sec)\nobjective\n{}\nvalue : {}".format(ab_time, ab_objective, get_value(unary_potential, ab_objective, pairwise_lamb)))
    print("greedy({}sec)\nobjective\n{}\nvalue : {}".format(greedy_time, greedy_objective,get_value(unary_potential, greedy_objective, pairwise_lamb)))
    print("unary\nobjective\n{}\nvalue : {}".format(unary_objective,get_value(unary_potential, unary_objective, pairwise_lamb)))

def test9():
    '''
    exp for rebutall
    '''
    pairwise_lamb = 0.3
    nworkers = 20
    ntasks = 4
    k = 1

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=pairwise_lamb)   

    unary_potential = np.random.random([nworkers, ntasks])
    mcf_objective = results2objective(results=mcf.solve(unary_potential), nworkers=nworkers, ntasks=ntasks)
    greedy_objective = greedy_max_matching_solver(array=unary_potential, plamb=pairwise_lamb, k=k)

    max_unary_potential = unary_potential
    max_deviation = get_value(unary_potential, greedy_objective, pairwise_lamb)-get_value(unary_potential, mcf_objective, pairwise_lamb)

    for _ in range(100000):
        unary_potential = np.random.random([nworkers, ntasks])
        mcf_objective = results2objective(results=mcf.solve(unary_potential), nworkers=nworkers, ntasks=ntasks)
        greedy_objective = greedy_max_matching_solver(array=unary_potential, plamb=pairwise_lamb, k=k)
        deviation =  get_value(unary_potential, greedy_objective, pairwise_lamb)-get_value(unary_potential, mcf_objective, pairwise_lamb)
        if deviation>max_deviation:
            max_deviation = deviation
            max_unary_potential = unary_potential

    unary_potential = max_unary_potential
    mcf_objective = results2objective(results=mcf.solve(unary_potential), nworkers=nworkers, ntasks=ntasks)
    greedy_objective = greedy_max_matching_solver(array=unary_potential, plamb=pairwise_lamb, k=k)

    print("unary potential : {}".format(unary_potential))
    print("mcf objective\n{}\nvalue : {}".format(mcf_objective, get_value(unary_potential, mcf_objective, pairwise_lamb)))
    print("greedy objective\n{}\nvalue : {}".format(greedy_objective,get_value(unary_potential, greedy_objective, pairwise_lamb)))

def test10():
    '''
    exp for rebutall
    '''
    pairwise_lamb = 1.0
    nworkers = 10
    ntasks = 2
    k = 1

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=pairwise_lamb)   

    
    hnworkers = nworkers//2
    unary_potential = np.array(hnworkers*[0.1, 0] + (nworkers-hnworkers)*[0, 0.2])
    unary_potential = np.reshape(unary_potential, [nworkers, ntasks])
    print(unary_potential)
    mcf_objective = results2objective(results=mcf.solve(unary_potential), nworkers=nworkers, ntasks=ntasks)
    greedy_objective = greedy_max_matching_solver(array=unary_potential, plamb=pairwise_lamb, k=k)

    print("unary potential\n: {}".format(unary_potential))
    print("mcf objective\n{}\nvalue : {}".format(mcf_objective, get_value(unary_potential, mcf_objective, pairwise_lamb)))
    print("greedy objective\n{}\nvalue : {}".format(greedy_objective,get_value(unary_potential, greedy_objective, pairwise_lamb)))

def test11():
    nclass = 64
    nworkers = 2*nclass
    ntasks = 5
    k = 1
    plamb = 0.1
    iterations = 100

    nsucess0 = 0
    nsucess1 = 0
    nsucess2 = 0
    nsucess3 = 0
    nsucess4 = 0
    nsucess5 = 0
    
    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=plamb)   
    unary_p = np.random.random([nworkers, ntasks])

    def results2objective(results, nw, nt):
        objective = np.zeros([nw, nt]) 
        for i,j in results:
            objective[i][j]=1
        return objective

    unary_p_t = np.zeros([nworkers, ntasks])
    unary_p_t2 = np.zeros([nworkers, ntasks])

    for i in range(nclass):
        for j in range(ntasks):
            if unary_p[2*i][j]<unary_p[2*i+1][j]:
                unary_p_t[2*i][j] = unary_p[2*i][j] + 0.5*plamb
                unary_p_t[2*i+1][j] = unary_p[2*i+1][j] - 0.5*plamb
            else:
                unary_p_t[2*i][j] = unary_p[2*i][j] - 0.5*plamb
                unary_p_t[2*i+1][j] = unary_p[2*i+1][j] + 0.5*plamb

    for i in range(nclass):
        for j in range(ntasks):
            if unary_p[2*i][j]<unary_p[2*i+1][j]:
                unary_p_t[2*i][j] = unary_p[2*i][j] + 0.5*plamb
                unary_p_t[2*i+1][j] = unary_p[2*i+1][j] - 0.5*plamb
            else:
                unary_p_t[2*i][j] = unary_p[2*i][j] - 0.5*plamb
                unary_p_t[2*i+1][j] = unary_p[2*i+1][j] + 0.5*plamb

    objective1 = results2objective(mcf.solve(unary_p_t), nworkers, ntasks)
    objective2 = results2objective(mcf.solve(unary_p), nworkers, ntasks)
    objective3 = results2objective(mcf.solve(unary_p_t2), nworkers, ntasks)
   
    DEM = DiscreteEnergyMinimize(ntasks, plamb)
    pairwise_term = np.zeros([nworkers, nworkers])
    for i in range(nclass):
        for j in range(i+1, nclass):
            pairwise_term[2*i][2*j] = 1
            pairwise_term[2*i+1][2*j] = 1
            pairwise_term[2*i][2*j+1] = 1
            pairwise_term[2*i+1][2*j+1] = 1

    #print("pairwise : {}".format(pairwise_term))
    ab_objective = DEM.solve(-unary_p, pairwise_term, k=k)
    #print("alpha beta objective : \n{}".format(ab_objective))

    results = list()
    for i in range(nworkers):
        random = np.argsort(-unary_p[i][:k])
        for j in random:
            results.append([i,j]) 
    tr_objective = results2objective(np.array(results), nworkers, ntasks)

    labels = np.reshape(np.tile(np.expand_dims(np.arange(nclass), axis=-1), (1,2)), [-1])
    gr_objective = greedy_max_matching_label_solver_iter(unary_p, plamb, labels, 10)

    energy0 = get_value_label(unary_p, objective1, plamb, labels)
    energy1 = get_value_label(unary_p, objective2, plamb, labels)
    energy2 = get_value_label(unary_p, objective3, plamb, labels)
    energy3 = get_value_label(unary_p, ab_objective, plamb, labels)
    energy4 = get_value_label(unary_p, tr_objective, plamb, labels)
    energy5 = get_value_label(unary_p, gr_objective, plamb, labels)

    min_energy = min(energy0, energy1, energy2, energy3, energy4, energy5)
    print(energy0, energy1, energy2, energy3, energy4, energy5)

def test12():
    nclass = 64
    nworkers = 2*nclass
    ntasks = 64
    k = 1
    plamb = 1.0
    iterations = 100

    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=plamb)   
    unary_p = np.random.random([nworkers, ntasks])

    def results2objective(results, nw, nt):
        objective = np.zeros([nw, nt]) 
        for i,j in results:
            objective[i][j]=1
        return objective

    unary_p_t = np.zeros([nworkers, ntasks])

    for i in range(nclass):
        for j in range(ntasks):
            if unary_p[2*i][j]<unary_p[2*i+1][j]:
                unary_p_t[2*i][j] = unary_p[2*i][j] + 0.5*plamb
                unary_p_t[2*i+1][j] = unary_p[2*i+1][j] - 0.5*plamb
            else:
                unary_p_t[2*i][j] = unary_p[2*i][j] - 0.5*plamb
                unary_p_t[2*i+1][j] = unary_p[2*i+1][j] + 0.5*plamb

    objective1 = results2objective(mcf.solve(unary_p_t), nworkers, ntasks)
    objective2 = results2objective(mcf.solve(unary_p), nworkers, ntasks)
   
    labels = np.reshape(np.tile(np.expand_dims(np.arange(nclass), axis=-1), (1,2)), [-1])

    energy1 = get_value_label(unary_p, objective1, plamb, labels)
    energy2 = get_value_label(unary_p, objective2, plamb, labels)

    print(energy1, energy2)
    
    for _ in range(iterations):
        gr_objective = greedy_max_matching_label_solver_iter(unary_p, plamb, labels, _)
        print("{} :{}".format(_, get_value_label(unary_p, gr_objective, plamb, labels)))

def test13():
    nclass = 4
    nworkers = 2*nclass
    ntasks = 5
    k = 1
    plamb = 1.0
   
    unary_p = np.random.random([nworkers, ntasks])
    labels = np.reshape(np.tile(np.expand_dims(np.arange(nclass), axis=-1), (1,2)), [-1])
    
    gr_objective = greedy_max_matching_label_solver_iter(unary_p, plamb, labels, 10)
    print("greedy objective :\n{}".format(gr_objective))
    print("{}".format(get_value_label(unary_p, gr_objective, plamb, labels)))

def test14():
    ''' test for solve_maxmatching_strict_intraclass
    '''
    nclass = 5
    nworkers = 2*nclass
    ntasks = 4

    unary_potential = np.random.random([nworkers, ntasks])
    print("unary_potential : \n{}".format(unary_potential))

    labels = np.arange(nclass)
    labels = np.concatenate([labels, labels], axis=0)

    results = solve_maxmatching_strict_intraclass(array=unary_potential, labels=labels, plamb=1.0)
    objective = np.zeros([nworkers, ntasks])
    for i,j in results: objective[i][j]=1

    print("labels : \n{}".format(labels))
    print("objective : \n{}".format(objective))
    print("objective : \n{}\n{}".format(objective[:nclass], objective[nclass:]))

def test15():
    ''' test for greedy_max_matching_label_solver_hungarian_initialization
    '''
    nclass = 5
    nworkers = 2*nclass
    ntasks = 5
    plamb = 1.0
    k=1

    unary_potential = np.random.random([nworkers, ntasks])

    labels = np.reshape(np.tile(np.expand_dims(np.arange(nclass), axis=-1), (1,2)), [-1])
    print("labels : \n{}".format(labels))

    hung = greedy_max_matching_label_solver_hungarian_initialization(array=unary_potential, plamb=plamb, labels=labels, k=k)
    iter_ = greedy_max_matching_label_solver_iter(unary_potential, plamb, labels, 1)
    iter10_ = greedy_max_matching_label_solver_iter(unary_potential, plamb, labels, 10)

    hung_value = get_value_label(potential=unary_potential, objective=hung, plamb=plamb, labels=labels)
    iter_value = get_value_label(potential=unary_potential, objective=iter_, plamb=plamb, labels=labels)
    iter10_value = get_value_label(potential=unary_potential, objective=iter10_, plamb=plamb, labels=labels)

    print("unary potential : \n {}\nhung objective(={}) :\n{}\niter(={}) :\n{}\niter10(={}) : \n{}".\
            format(unary_potential, hung_value, hung, iter_value, iter_, iter10_value, iter10_))

def test16():
    ''' test for greedy_max_matching_label_solver_k, greedy_max_matching_label_solver_k_iter
    '''
    nclass = 5
    nworkers = 2*nclass
    ntasks = 5
    plamb = 1.0
    k=2

    unary_potential = np.random.random([nworkers, ntasks])

    labels = np.reshape(np.tile(np.expand_dims(np.arange(nclass), axis=-1), (1,2)), [-1])
    print("labels : \n{}".format(labels))

    hung = greedy_max_matching_label_solver_hungarian_initialization(array=unary_potential, plamb=plamb, labels=labels, k=k)
    iter_, _ = greedy_max_matching_label_solver_k(array=unary_potential, plamb=plamb, labels=labels, k=k)
    iter10_ = greedy_max_matching_label_solver_k_iter(array=unary_potential, plamb=plamb, labels=labels, k=k, niter=10)

    hung_value = get_value_label(potential=unary_potential, objective=hung, plamb=plamb, labels=labels)
    iter_value = get_value_label(potential=unary_potential, objective=iter_, plamb=plamb, labels=labels)
    iter10_value = get_value_label(potential=unary_potential, objective=iter10_, plamb=plamb, labels=labels)

    print("unary potential : \n {}\nhung objective(={}) :\n{}\niter(={}) :\n{}\niter10(={}) : \n{}".\
            format(unary_potential, hung_value, hung, iter_value, iter_, iter10_value, iter10_))


def test17():
    ''' test for SolveMaxMatchingHungarian
    '''
    nsclass = 4
    ndata_per_class = 3
    ntasks = 6
    ndata = nsclass*ndata_per_class
    plamb = 0.1

    unary = np.random.random([ndata, ntasks])
    unary_class = np.mean(np.reshape(unary, [nsclass, -1, ntasks]), axis=1)

    mcf = SolveMaxMatching(nworkers=nsclass, ntasks=ntasks, k=ndata_per_class, pairwise_lamb=plamb)
    hung = SolveMaxMatchingHungarian(nworkers=ndata_per_class, ntasks=ndata_per_class, k=1)

    results_summary = list()
    for i in range(nsclass):
        results_summary.append(list())

    results = mcf.solve(unary_class)
    for i,j in results:
        results_summary[i].append(j)

    objective = np.zeros([ndata, ntasks], dtype=np.float32) # [nbatch, d]
    for i in range(nsclass):
        unary_tmp = np.zeros([ndata_per_class, ndata_per_class])
        for j1 in range(ndata_per_class):
            for j2 in range(ndata_per_class):
                unary_tmp[j1][j2] = unary[ndata_per_class*i+j1][results_summary[i][j2]]
        results = hung.solve(unary_tmp)
        for a, b in results:
            objective[ndata_per_class*i+a][results_summary[i][b]] = 1

    print("unary : \n{}".format(unary))
    print("unary_class : \n{}".format(unary_class))
    print("results_summary : \n{}".format(results_summary))
    print("objective : \n{}".format(objective))

def test18():
    '''debugging for solve_maxmatching_soft_intraclass
    '''
    nworkers = 5
    ntasks = 4
    labels = [0,1,2,1,2]
    plamb1 = 0.1
    plamb2 = 1.0
    value = 10000

    array = np.random.random([nworkers, ntasks])
    print("array :\n{}".format(array))
    print("labels : {}".format(labels))
    subsets = plabel2subset(labels=labels)
    print("subsets : {}".format(subsets))
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
    print("occupy : {}".format(occupy))
    print("nsubset_list : {}".format(nsubset_list))

    source = 0
    sink = 1 + nworkers + nsubsets*ntasks + ntasks
    pcost1 = int(plamb1*value)
    pcost2 = int(plamb2*value)
    
    # source [0]
    # workers [1:nworkers], [1+w_idx]
    # subsets [1+nworkers:nworkers+ntasks*nsubsets], [1+nworkers+ntasks*s_idx+w_idx]
    # tasks [1+nworkers+ntasks*nsubsets:nworkers+ntasks*nsubsets+ntasks]
    supplies = [nworkers]+(sink-1)*[0]+[-nworkers]

    start_nodes = list()
    end_nodes = list()
    capacities = list()
    costs = list()

    print("source to workers")
    for w_idx in range(nworkers):
        start_nodes.append(source)
        end_nodes.append(w_idx+1)
        capacities.append(1)
        costs.append(0)
        print(" {} to {}, capacity {} cost {}".format(source, 1+w_idx, 1, 0))

    print("workers2subsets or tasks")
    for w_idx in range(nworkers):
        if occupy[w_idx]==-1: # to tasks
            for t_idx in range(ntasks):
                start_nodes.append(1 + w_idx) # workers
                end_nodes.append(1 + nworkers + nsubsets*ntasks + t_idx) # tasks
                capacities.append(1)
                costs.append(array_p[w_idx][t_idx])
                print(" {} to {}, capacity {} cost{}".format(1 + w_idx, 1 + nworkers + nsubsets*ntasks+t_idx, 1, array_p[w_idx][t_idx]))
        else:
            s_idx = occupy[w_idx]
            for t_idx in range(ntasks):
                start_nodes.append(1 + w_idx) # workers
                end_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
                capacities.append(1)
                costs.append(array_p[w_idx][t_idx])
                print(" {} to {}, capacity {} cost{}".format(1+w_idx, 1 + nworkers + s_idx*ntasks+t_idx, 1, array_p[w_idx][t_idx]))

    print("subsets to tasks")
    # subsets2tasks
    for s_idx in range(nsubsets):
        nworkers_subset = nsubset_list[s_idx]
        for t_idx in range(ntasks):
            for w_idx in range(nworkers_subset):
                start_nodes.append(1 + nworkers + s_idx*ntasks + t_idx) # subsets
                end_nodes.append(1 + nworkers + nsubsets*ntasks + t_idx) # tasks
                capacities.append(1)
                costs.append(w_idx*pcost1)
                print(" {} to {} capacity {} cost {}".format(1+nworkers+s_idx*ntasks+t_idx, 1+nworkers+nsubsets*ntasks+t_idx, 1, w_idx*pcost1))

    print("tasks to sink")
    for t_idx in range(ntasks):
        for w_idx in range(nworkers):
            start_nodes.append(1 + nworkers + ntasks*nsubsets + t_idx)
            end_nodes.append(sink)
            capacities.append(1)
            costs.append(w_idx*pcost2)
            print(" {} to {} capacity {} cost {}".format(1+nworkers+ntasks*nsubsets+t_idx, sink, 1, w_idx*pcost2))

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

def test19():
    '''timecompare
    solve_maxmatching_soft_intraclass_multiselect
    SolveMaxMatching

    Results -
        mcf : 0.8217835187911987 sec
        smsi : 0.08805792331695557 sec
    '''
    nworkers = 128 
    sk = 2
    d = 32 
    k = 2 
    max_label = 32
    plamb = 0.1

    ntrial = 20
    mcf_time = 0
    smsi_time = 0

    for _ in range(ntrial):
        plabel = np.random.randint(max_label, size=nworkers)

        unary = np.random.random([nworkers, d**k])
        unary1 = np.random.random([nworkers, d])
        unary2 = np.random.random([nworkers, d])
        mcf = SolveMaxMatching(nworkers=nworkers, ntasks=d**k, k=sk, pairwise_lamb=plamb)

        mcf_start_time = time.time() 
        results = mcf.solve(unary)
        mcf_end_time = time.time() 

        smsi_start_time = time.time()
        solve_maxmatching_soft_intraclass_multiselect(array=unary1, k=sk, labels=plabel, plamb1=plamb, plamb2=0.0)
        solve_maxmatching_soft_intraclass_multiselect(array=unary2, k=sk, labels=plabel, plamb1=plamb, plamb2=0.0)
        smsi_end_time = time.time()
        
        mcf_time += mcf_end_time-mcf_start_time
        smsi_time += smsi_end_time-smsi_start_time

    mcf_time/=ntrial
    smsi_time/=ntrial
    print("mcf : {} sec".format(mcf_time))
    print("smsi : {} sec".format(smsi_time))

def test20():
    '''time complexity of solve_maxmatching_soft_intraclass_multiselect
    Resutls -
        n_c : time
        64 : 0.02199897766113281 sec
        128 : 0.04322974681854248 sec
        256 : 0.08782792091369629 sec
        512 : 0.18317224979400634 sec
        d : time
        32 : 0.08886291980743408 sec
        64 : 0.17593204975128174 sec
        128 : 0.3502910614013672 sec
        256 : 0.6889388084411621 sec
        512 : 1.376410722732544 sec
    '''
    sk = 2
    d = 32 
    k = 2 
    max_label = 8
    plamb = 0.1

    ntrial = 20

    nworkers_list = [64, 128, 256, 512]
    smsi_time_list = list()
    
    for nworkers in nworkers_list:
        smsi_time = 0
        for idx in range(ntrial):
            plabel = np.random.randint(max_label, size=nworkers)

            unary = np.random.random([nworkers, d])
            smsi_start_time = time.time()
            solve_maxmatching_soft_intraclass_multiselect(array=unary, k=sk, labels=plabel, plamb1=plamb, plamb2=0.0)
            smsi_end_time = time.time()
            
            smsi_time += smsi_end_time-smsi_start_time
        smsi_time/=ntrial
        smsi_time_list.append(smsi_time)

    print("n_c : time")
    for idx in range(len(nworkers_list)):
        print("{} : {} sec".format(nworkers_list[idx], smsi_time_list[idx]))

    nworkers = 256
    d_list = [32, 64, 128, 256, 512]
    smsi_time_list = list()

    for d in d_list:
        smsi_time = 0
        for idx in range(ntrial):
            plabel = np.random.randint(max_label, size=nworkers)

            unary = np.random.random([nworkers, d])
            smsi_start_time = time.time()
            solve_maxmatching_soft_intraclass_multiselect(array=unary, k=sk, labels=plabel, plamb1=plamb, plamb2=0.0)
            smsi_end_time = time.time()
            
            smsi_time += smsi_end_time-smsi_start_time
        smsi_time/=ntrial
        smsi_time_list.append(smsi_time)

    print("d : time")
    for idx in range(len(d_list)):
        print("{} : {} sec".format(d_list[idx], smsi_time_list[idx]))

def test21():
    '''timecompare
    '''
    nworkers = 6
    sk = 1
    d = 4
    max_label = 2

    plabel = np.random.randint(max_label, size=nworkers)
    unary1 = np.random.random([nworkers, d])
    print(plabel)
    print(unary1)
    print(solve_maxmatching_soft_intraclass_multiselect_greedy(array=unary1, k=sk, labels=plabel, plamb1=1000.0, plamb2=0.0))

if __name__=="__main__":
    test21()



