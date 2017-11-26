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

tf.app.flags.DEFINE_string(
    'dataset_dir', "/world/data-c9/liubofang/dataset_original/cifar10",
    'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'gpus', "6,", 'which gpu to used.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 10,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_string(
    'restore_ckpt', "", 'Restore checkpoint.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer(
    'try_num', 0, 'Try num flag.')
tf.app.flags.DEFINE_string(
    'working_root', "/world/data-c9/liubofang/training/classification/cifar10",
    'Working root folder.')
tf.app.flags.DEFINE_float('base_lr', 0.01, 'Initial learning rate.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # set up TF environment
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    num_gpus = len(FLAGS.gpus.split(','))
    # save prefix
    prefix = '%s/%s/%d' % (FLAGS.working_root, FLAGS.model_name, FLAGS.try_num)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    with tf.Graph().as_default():
        # Create a dataset provider that loads data from the dataset #
        dataset = dataset_factory.get_dataset("cifar10", "train", FLAGS.dataset_dir)
        # Create a model
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=dataset.num_classes,
            weight_decay=0.00004,
            is_training=True)
        with tf.device("/cpu:0"):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20 * FLAGS.batch_size,
                common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])

            train_image_size = FLAGS.train_image_size or network_fn.default_image_size

            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name,
                is_training=True)
            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            labels = slim.one_hot_encoding(labels, dataset.num_classes)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * num_gpus)
        images, labels = batch_queue.dequeue()
        logits, end_points = network_fn(images)
        # Create loss
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        # Create global_step and lr
        global_step = slim.create_global_step()
        learning_rate = get_learning_rate("exponential", FLAGS.base_lr,
                                          global_step, decay_steps=10000)
        # Create optimizer
        optimizer = get_optimizer('sgd', learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # Create sess
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                gpu_options=tf.GPUOptions(allow_growth=True)
                                                ))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        # If resotre checkpoint
        if FLAGS.restore_ckpt:
            variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, arg_dict['restore_ckpt'])
            print ('Resume-trained model restored from: %s' % arg_dict['restore_ckpt'])
        # start all queue
        tf.train.start_queue_runners(sess=sess)
        print ("Start to training...")
        start_time = time.time()
        while True:
            with tf.device('/gpu:0'):
                _, ploss, step, lr = sess.run([train_op, loss, global_step, learning_rate])
                if step % 10 == 0:
                    end_time = time.time()
                    cost_time, start_time = end_time - start_time, end_time
                    sample_per_sec = int(10 * FLAGS.batch_size / cost_time)
                    sec_per_step = cost_time / 10.0
                    print ('[%s] epochs: %d, step: %d, lr: %f, loss: %.4f, '
                           'sample/s: %d, sec/step: %.3f' % (
                               datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                               -1, step, lr, ploss,
                               sample_per_sec, sec_per_step))
            if step % 1024 == 0:
                checkpoint_path = os.path.join(prefix, 'model.ckpt')
                saver.save(sess, checkpoint_path)
                print ('Saved checkpoint to %s' % checkpoint_path)
        checkpoint_path = os.path.join(prefix, 'model.ckpt')
        saver.save(sess, checkpoint_path)
        print ('\nReview training parameter:\n%s\n' % (str(arg_dict)))
        print ('Saved checkpoint to %s' % checkpoint_path)
        print ('Bye Bye!')


if __name__ == "__main__":
    tf.app.run()
