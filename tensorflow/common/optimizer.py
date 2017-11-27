import tensorflow as tf

def get_optimizer(optimizer, learning_rate, **kwargs):
    """Configures the optimizer used for training.

    Args:
        optimizer: optimizer name
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        Unsupport optimizer name
    """
    if optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=kwargs.get("adadelta_rho", 0.95),
            epsilon=kwargs.get("opt_epsilon", 1e-08))
    elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=kwargs.get("adagrad_initial_accumulator_value", 0.1))
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=kwargs.get("adam_beta1", 0.9),
            beta2=kwargs.get("adam_beta2", 0.999),
            epsilon=kwargs.get("opt_epsilon", 1.0)) # 1e-8
    elif optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=kwargs.get("ftrl_learning_rate_power", -0.5),
            initial_accumulator_value=kwargs.get("ftrl_initial_accumulator_value", 0.1),
            l1_regularization_strength=kwargs.get("ftrl_l1", 0.0),
            l2_regularization_strength=kwargs.get("ftrl_l2", 0.0))
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=kwargs["momentum"],
            name='Momentum')
    elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=kwargs.get("rmsprop_decay", 0.9),
            momentum=kwargs.get("rmsprop_momentum", 0.9), # 0.0
            epsilon=kwargs.get("opt_epsilon", 1.0)) # 1e-10
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', optimizer)
    return optimizer
