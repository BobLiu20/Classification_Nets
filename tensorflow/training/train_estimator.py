# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from common.learning_rate import get_learning_rate
from common.optimizer import get_optimizer
from common.average_gradients import average_gradients

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string(
    'dataset_dir', "/world/data-c9/liubofang/dataset_original/cifar10",
    'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'gpus', "7,", 'which gpu to used. support multi gpu')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 10,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'batch_size', 256, 'The number of samples in each batch for each gpu')
tf.app.flags.DEFINE_integer(
    'train_image_size', 32, 'Train image size')
tf.app.flags.DEFINE_string(
    'model_name', 'cifarnet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer(
    'try_num', 0, 'Try num flag.')
tf.app.flags.DEFINE_string(
    'working_root', "/world/data-c9/liubofang/training/classification",
    'Working root folder.')
tf.app.flags.DEFINE_float('base_lr', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

FLAGS = tf.app.flags.FLAGS


def input_fn(num_gpus, dataset_split="train"):
    batch_size = FLAGS.batch_size * num_gpus
    # Create a dataset provider that loads data from the dataset #
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, dataset_split, FLAGS.dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        FLAGS.model_name,
        is_training=True)
    image = image_preprocessing_fn(
        image, FLAGS.train_image_size, FLAGS.train_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2 * num_gpus)
    images, labels = batch_queue.dequeue()
    return images, labels


def model_fn(features, labels, mode, params):
    # network
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=params['num_classes'],
        weight_decay=0.00004,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    # if predict. Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits, end_points = network_fn(features)
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"output": logits})
    # Create global_step and lr
    global_step = tf.train.get_global_step()
    learning_rate = get_learning_rate("exponential", FLAGS.base_lr,
                                      global_step, decay_steps=10000)
    # Create optimizer
    optimizer = get_optimizer(FLAGS.optimizer, learning_rate)
    # Multi GPU support
    features_split = tf.split(features, len(params['gpus_list']), axis=0)
    labels_split = tf.split(labels, len(params['gpus_list']), axis=0)
    tower_grads = []
    eval_logits = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(len(params['gpus_list'])):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("classification", i)) as scope:
                    # model and loss
                    logits, end_points = network_fn(features_split[i])
                    tf.losses.softmax_cross_entropy(labels_split[i], logits)
                    update_ops = tf.get_collection(
                        tf.GraphKeys.UPDATE_OPS, scope)
                    updates_op = tf.group(*update_ops)
                    with tf.control_dependencies([updates_op]):
                        losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                    # reuse var
                    tf.get_variable_scope().reuse_variables()
                    # grad compute
                    grads = optimizer.compute_gradients(total_loss)
                    tower_grads.append(grads)
                    # for eval metric ops
                    eval_logits.append(logits)
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optimizer.apply_gradients(
        grads, global_step=global_step)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)
    # Create eval metric ops
    _predictions = tf.argmax(tf.concat(eval_logits, 0), 1)
    _labels = tf.argmax(labels, 1)
    eval_metric_ops = {
        "acc": slim.metrics.streaming_accuracy(_predictions, _labels)}
    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def main(_):
    # set up TF environment
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    gpus_list = FLAGS.gpus.split(',')
    # save prefix
    prefix = '%s/%s/%s/%d' % (FLAGS.working_root, FLAGS.dataset_name,
                              FLAGS.model_name, FLAGS.try_num)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # start
    model_params = {"num_classes": 10, "gpus_list": gpus_list}
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        model_dir=prefix,
        log_step_count_steps=100,
        save_checkpoints_secs=600,
        session_config=tf.ConfigProto(allow_soft_placement=True,
                                      gpu_options=tf.GPUOptions(allow_growth=True)))
    nn = tf.estimator.Estimator(
        model_fn=model_fn, params=model_params, config=run_config)
    nn.train(input_fn=lambda: input_fn(
        len(gpus_list)), steps=None, max_steps=None)


if __name__ == "__main__":
    tf.app.run()
