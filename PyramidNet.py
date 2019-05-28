from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import os


class PyramidNet:
    def __init__(self, config, input_shape, num_classes, weight_decay, data_format):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        assert data_format in ['channels_last', 'channels_first']
        self.data_format = data_format

        self.config = config
        self.is_bottleneck = config['is_bottleneck']
        self.block_list = config['residual_block_list']
        self.downsampling_list = [np.sum(self.block_list[:i+1]) for i in range(len(self.block_list))]
        alpha = config['alpha']
        total_units = np.sum(self.block_list)
        self.filters_list = [config['init_conv_filters']+(i+1)*alpha//total_units for i in range(total_units)]
        self.global_step = tf.train.get_or_create_global_step()
        self.is_training = True

        self._define_inputs()
        self._build_graph()
        self._init_session()

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.input_shape)
        self.images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):
        with tf.variable_scope('before_split'):
            conv1_1 = self._conv_bn_activation(
                bottom=self.images,
                filters=self.config['init_conv_filters'],
                kernel_size=self.config['init_conv_kernel_size'],
                strides=self.config['init_conv_strides'],
                )
            pool1 = self._max_pooling(
                bottom=conv1_1,
                pool_size=self.config['init_pooling_pool_size'],
                strides=self.config['init_pooling_strides'],
                name='pool1'
            )
        if self.is_bottleneck:
            stack_residual_unit_fn = self._residual_bottleneck
            filters_list = [self.config['init_conv_filters']]
            filters_list.extend([var*4 for var in self.filters_list])
        else:
            stack_residual_unit_fn = self._residual_block
            filters_list = [self.config['init_conv_filters']]
            filters_list.extend(self.filters_list)
        with tf.variable_scope('split'):
            residual_block = pool1
            j = 0
            for i in range(len(self.filters_list)):
                if i+1 in self.downsampling_list:
                    j += 1
                    if i+1 != 1:
                        residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 2, 'block'+str(j+1)+'_unit'+str(i+1), filters_list[i])
                    else:
                        residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 1, 'block'+str(j+1)+'_unit'+str(i+1), filters_list[i])
                else:
                    residual_block = stack_residual_unit_fn(residual_block, self.filters_list[i], 2, 'block'+str(j+1)+'_unit'+str(i+1), filters_list[i])

            residual_block = tf.nn.relu(residual_block)
        with tf.variable_scope('after_spliting'):
            bn = self._bn(residual_block)
            relu = tf.nn.relu(bn)
        with tf.variable_scope('final_dense'):
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(residual_block, axis=axes, keepdims=False, name='global_pool')
            final_dense = tf.layers.dense(global_pool, self.num_classes, name='final_dense')
        with tf.variable_scope('optimizer'):
            self.logit = tf.nn.softmax(final_dense, name='logit')
            self.classifer_loss = tf.losses.softmax_cross_entropy(self.labels, final_dense, label_smoothing=0.1, reduction=tf.losses.Reduction.MEAN)
            self.l2_loss = self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
            )
            total_loss = self.classifer_loss + self.l2_loss
            lossavg = tf.train.ExponentialMovingAverage(0.9, name='loss_moveavg')
            lossavg_op = lossavg.apply([total_loss])
            with tf.control_dependencies([lossavg_op]):
                self.total_loss = tf.identity(total_loss)
            var_list = tf.trainable_variables()
            varavg = tf.train.ExponentialMovingAverage(0.9, name='var_moveavg')
            varavg_op = varavg.apply(var_list)
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
            train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([update_ops, lossavg_op, varavg_op, train_op])
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(final_dense, 1), tf.argmax(self.labels, 1)), tf.float32), name='accuracy'
            )

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def train_one_batch(self, images, labels, lr, sess=None):
        self.is_training = True
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        _, loss, acc = sess_.run([self.train_op, self.total_loss, self.accuracy],
                                 feed_dict={
                                     self.images: images,
                                     self.labels: labels,
                                     self.lr: lr
                                 })
        return loss, acc

    def validate_one_batch(self, images, labels, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        logit, acc = sess_.run([self.logit, self.accuracy], feed_dict={
                                     self.images: images,
                                     self.labels: labels,
                                     self.lr: 0.
                                 })
        return logit, acc

    def test_one_batch(self, images, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        logit = sess_.run([self.logit], feed_dict={
                                     self.images: images,
                                     self.lr: 0.
                                 })
        return logit

    def save_weight(self, mode, path, sess=None):
        assert(mode in ['latest', 'best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        saver = self.saver if mode == 'latest' else self.best_saver
        saver.save(sess_, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, mode, path, sess=None):
        assert(mode in ['latest', 'best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        saver = self.saver if mode == 'latest' else self.best_saver
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess_, path)
            print('load', mode, 'model in', path, 'successfully')
        else:
            raise FileNotFoundError('Not Found Model File!')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _bn_activation_conv(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        bn = self._bn(bottom)
        if activation is not None:
            bn = activation(bn)
        conv = tf.layers.conv2d(
            inputs=bn,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        return conv

    def _residual_block(self, bottom, filters, strides, scope, input_filters):
        with tf.variable_scope(scope):
            conv = self._bn_activation_conv(bottom, filters, 3, strides, None)
            conv = self._bn_activation_conv(conv, filters, 3, 1)
            shutcut = self._avg_pooling(bottom, strides, strides, 'avg_pool')
            if self.data_format == 'channels_last':
                shutcut = tf.pad(shutcut, [[0, 0], [0, 0], [0, 0], [0, filters-input_filters]])
            else:
                shutcut = tf.pad(shutcut, [[0, 0],  [0, filters-input_filters], [0, 0], [0, 0]])
            return conv + shutcut

    def _residual_bottleneck(self, bottom, filters, strides, scope, input_filters):
        with tf.variable_scope(scope):
            conv = self._bn_activation_conv(bottom, filters, 1, 1, None)
            conv = self._bn_activation_conv(conv, filters, 3, strides)
            conv = self._bn_activation_conv(conv, filters*4, 1, 1)
            if strides != 1:
                shutcut = self._avg_pooling(bottom, strides, strides, 'avg_pool')
                shutcut = tf.pad(shutcut, [[0, 0], [0, 0], [0, 0], [0, filters*4-input_filters]])
            else:
                shutcut = tf.pad(bottom, [[0, 0], [0, 0], [0, 0], [0, filters*4-input_filters]])
            return conv + shutcut

    def _max_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )


