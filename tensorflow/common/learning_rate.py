import tensorflow as tf


def get_learning_rate(lr_type, base_lr, global_step, decay_steps=10000):
    """Configures the learning rate.

    Args:
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """

    if lr_type == 'exponential':
        return tf.train.exponential_decay(base_lr,
                                          global_step,
                                          decay_steps,
                                          0.8,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif lr_type == 'fixed':
        return tf.constant(base_lr, name='fixed_learning_rate')
    elif lr_type == 'polynomial':
        return tf.train.polynomial_decay(base_lr,
                                         global_step,
                                         decay_steps,
                                         0.00001,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('lr_type [%s] was not recognized',
                         lr_type)
