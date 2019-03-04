import tensorflow as tf

def get_shape(t):
    '''get the shape of tensor as list
    Args:
        t - tensor
    Return:
        list - shape of t
    '''
    return t.get_shape().as_list()

def vars_info_vl(var_list): 
    return "    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in var_list])

def vars_info(string):
    '''print variables in collection named string'''
    return "Collection name %s\n"%string+vars_info_vl(tf.get_collection(string))

def get_init_vars(sess):
    init_vars = []
    for var in tf.global_variables():
        try: sess.run(var)
        except tf.errors.FailedPreconditionError: continue
        init_vars.append(var)
    return init_vars

def get_uninit_vars(sess):
    uninit_vars = []
    for var in tf.global_variables():
        try : sess.run(var)
        except tf.errors.FailedPreconditionError: uninit_vars.append(var)
    return uninit_vars

def count_vars(var_list):
    '''count the # of vars in var_list'''
    count = 0
    for var in var_list:
        var_shape = get_shape(var)
        var_size = 1
        for size in var_shape:
            var_size*=size
        count+=var_size
    return count

