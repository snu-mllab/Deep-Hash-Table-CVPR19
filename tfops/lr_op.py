import tensorflow as tf

def stair_decay(initial_lr, decay_steps, decay_rate, initial_step=0):
    '''
    Args:
        initial_lr - float
        decay_steps - int
        decay_rate - float
        initial_step - int
    Return : 
        learing_rate - self decaying
            initial_lr*decay_rate^int(global_step/decay_steps)
        update_step_op - tf op
            add 1 global step
    '''
    global_step = tf.Variable(initial_step, trainable=False)
    update_step_op = tf.assign_add(global_step, 1)
    return tf.train.exponential_decay(
                learning_rate=initial_lr,\
                global_step=global_step,\
                decay_steps=decay_steps,\
                decay_rate=decay_rate,\
                staircase=True), update_step_op

def piecewise_decay(boundaries, values, initial_step = 0):
    '''
    Args:
        initial_step - int defaults to be 0
        boundaries - list with int 
        values - list with float
    Return : 
        learing_rate - self decaying
            
        update_step_op - tf op
            add 1 global step
    '''
    global_step = tf.Variable(initial_step, name='global_step', trainable=False)
    update_step_op = tf.assign_add(global_step, 1)
    return tf.train.piecewise_constant(global_step, boundaries, values), update_step_op

DECAY_DICT = {
            'stair' : stair_decay,
            'piecewise' : piecewise_decay
            }

DECAY_PARAMS_DICT =\
    {
    'stair' : 
        {
            128 : {
                'a1':  {'initial_lr' : 1e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a6' : {'initial_lr' : 3e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a7' : {'initial_lr' : 1e-2, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'b1':  {'initial_lr' : 1e-5, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'b2' : {'initial_lr' : 3e-5, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'b3' : {'initial_lr' : 1e-4, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'b4' : {'initial_lr' : 3e-4, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'b5' : {'initial_lr' : 1e-3, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'b6' : {'initial_lr' : 3e-3, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'b7' : {'initial_lr' : 1e-2, 'decay_steps' : 20000, 'decay_rate' : 0.3},
                'c1':  {'initial_lr' : 1e-5, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                'c2' : {'initial_lr' : 3e-5, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                'c3' : {'initial_lr' : 1e-4, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                'c4' : {'initial_lr' : 3e-4, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                'c5' : {'initial_lr' : 1e-3, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                'c6' : {'initial_lr' : 3e-3, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                'c7' : {'initial_lr' : 1e-2, 'decay_steps' : 8000, 'decay_rate' : 0.3},
                },
            512 :
                {
                'a1' : {'initial_lr' : 1e-5, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-5, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-4, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-4, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-3, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                'a6' : {'initial_lr' : 3e-3, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                'a7' : {'initial_lr' : 1e-2, 'decay_steps' : 80000, 'decay_rate' : 0.3},
                },
            1024 :
                {
                'a1' : {'initial_lr' : 1e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a6' : {'initial_lr' : 3e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a7' : {'initial_lr' : 1e-2, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                }
        }
    }

