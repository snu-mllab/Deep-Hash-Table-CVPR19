import tensorflow as tf

def get_train_op(optimizer, loss, lr, var_list=tf.trainable_variables()):
    '''
    Args:
        optimizer - tf optimizer
                ex) tf.train.AdamOptimizer
        loss - a tensor
        lr - float
            learning rate
        var_list - list of tensors
    Return:
        train_op
    '''
    return optimizer(lr).minimize(loss=loss, var_list=var_list)

def get_train_op_v2(optimizer, loss, var_list=tf.trainable_variables()):
    '''
    Args:
        optimizer - tf optimizer
                ex) tf.train.AdamOptimizer(1e-4) 
        loss - a tensor
        var_list - list of tensors
    Return:
        train_op
    '''
    return optimizer.minimize(loss=loss, var_list=var_list)

def get_multi_train_op(optimizer, loss, lr_list, vl_list):
    '''
    Args:
        optimizer - tf optimizer
                ex) tf.train.AdamOptimizer 
        loss - a tensor
        lr_list - learning rate list
        vl_list - list of variable list
    Return:
        train_op
    '''
    assert len(lr_list)==len(vl_list), "The length of lr_list, and vl_list should be same but %d and %d"%(len(lr_list), len(vl_list))

    vl_true_list, lr_true_list = list(), list()

    for idx in range(len(lr_list)):
        if len(vl_list[idx])==0: continue
        vl_true_list.append(vl_list[idx])
        lr_true_list.append(lr_list[idx])

    vl_list, lr_list = vl_true_list, lr_true_list

    nlist = len(lr_list)
    opt_list = list()
    grad_list = list()
    train_op_list = list()
   
    def list_summer(list_):
        v = list_[0]
        for i in range(1, len(list_)):
            v=v+list_[i]
        return v
    grads = tf.gradients(loss, list_summer(vl_list))

    for i in range(nlist):
        opt_list.append(optimizer(lr_list[i]))

    start = 0
    for i in range(nlist):
        grad_list.append(grads[start:start+len(vl_list[i])]) 
        train_op_list.append(opt_list[i].apply_gradients(zip(grad_list[i], vl_list[i])))
        start+=len(vl_list[i])
    train_op = tf.group(*train_op_list)

    return train_op 

