from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


class Inceptionv3:

    def __init__(self, input_shape, num_classes, weight_decay, keep_prob, data_format, config=None):

        assert(data_format in ['channels_last', 'channels_first'])
        if data_format == 'channels_last':
            assert((input_shape[0] >= 43) & (input_shape[1] >= 43))
            self.input_shape = input_shape
            self.output_shape = np.array([input_shape[0], input_shape[1]], dtype=np.int32)
        else:
            assert((input_shape[1] >= 43) & (input_shape[2] >= 43))
            self.input_shape = input_shape
            self.output_shape = np.array([input_shape[1], input_shape[2]], dtype=np.int32)

        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.prob = 1. - keep_prob
        self.data_format = data_format
        if config is not None:
            self.is_SENet = config['is_SENet']
            if config['is_SENet']:
                self.reduction = config['reduction']
        else:
            self.is_SENet = False

        self.global_step = tf.train.get_or_create_global_step()
        self.is_training = True

        self._define_input()
        self._build_graph()
        self._init_session()

    def _define_input(self):
        shape = [None]
        shape.extend(self.input_shape)
        self.images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):
        with tf.variable_scope('stem'):
            conv1_1 = self._conv_bn_activation(self.images, 32, 3, 2, 'valid')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'valid', 2)
            conv1_2 = self._conv_bn_activation(conv1_1, 32, 3, 1, 'valid')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'valid', 1)
            conv1_3 = self._conv_bn_activation(conv1_2, 64, 3, 1, 'same')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'same', 1)
            pool1 = self._max_pooling(conv1_3, 3, 2, 'valid', 'pool1')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'valid', 2)

            conv2_1 = self._conv_bn_activation(pool1, 80, 3, 1, 'valid')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'valid', 1)
            conv2_2 = self._conv_bn_activation(conv2_1, 192, 3, 2, 'valid')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'valid', 2)
            conv2_3 = self._conv_bn_activation(conv2_2, 288, 3, 1, 'valid')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'valid', 1)
        with tf.variable_scope('inception_1'):
            inception1_1 = self._inception_block1(conv2_3, [64, 48, 96, 32], 'inception1_1')
            inception1_2 = self._inception_block1(inception1_1, [64, 48, 96, 64], 'inception1_2')
            inception1_3 = self._inception_block1(inception1_2, [64, 48, 96, 64], 'inception1_3')
            inception1_reduction = self._grid_size_reduction1(inception1_3, 'inception1_reduction')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'same', 2)
            self.auxiliary_outshape = self.output_shape
        with tf.variable_scope('inception_2'):
            inception2_1 = self._inception_block2(inception1_reduction, [192, 128], 'inception2_1')
            inception2_2 = self._inception_block2(inception2_1, [192, 160], 'inception2_2')
            inception2_3 = self._inception_block2(inception2_2, [192, 160], 'inception2_3')
            inception2_4 = self._inception_block2(inception2_3, [192, 192], 'inception2_4')
            inception2_reduction = self._grid_size_reduction2(inception2_4, 'inception2_reduction')
            self.output_shape = self._compute_output_shape(self.output_shape, 3, 'same', 2)
        with tf.variable_scope('inception_3'):
            inception3_1 = self._inception_block3(inception2_reduction, [320, 384, 448, 192], 'inception3_1')
            inception3_2 = self._inception_block3(inception3_1, [320, 384, 448, 192], 'inception3_2')
        with tf.variable_scope('classifier'):
            global_pool = self._max_pooling(inception3_2, self.output_shape.astype(np.int32).tolist(), 1, 'valid', 'global_pool')
            dropout = self._dropout(global_pool, 'dropout')
            final_dense = self._conv_bn_activation(dropout, self.num_classes, 1, 1, 'same', None)
            logit = tf.squeeze(final_dense, name='logit')
            self.logit = tf.nn.softmax(logit, name='softmax')
        with tf.variable_scope('auxilary'):
            auxiliary_avgpool = self._avg_pooling(inception1_reduction, 5, 3, 'same', 'auxiliary_avgpool')
            self.auxiliary_outshape = self._compute_output_shape(self.auxiliary_outshape, 5, 'same', 3)
            auxiliary_conv1x1 = self._conv_bn_activation(auxiliary_avgpool, 128, 1, 1, 'same')
            self.auxiliary_outshape = self._compute_output_shape(self.auxiliary_outshape, 1, 'same', 1)
            new_channels = np.prod(self.auxiliary_outshape).astype(np.int32) * 128
            auxiliary_conv1x1_reshape = tf.reshape(auxiliary_conv1x1, [-1, 1, 1, new_channels], name='auxiliary_conv1x1_reshape')
            auxiliary_final_dense = self._conv_bn_activation(auxiliary_conv1x1_reshape, self.num_classes, 1, 1, 'valid', None)
            auxiliary_logit = tf.squeeze(auxiliary_final_dense, name='logit')
            auxiliary_loss = 0.4 * tf.losses.softmax_cross_entropy(self.labels, auxiliary_logit, label_smoothing=0.1, reduction=tf.losses.Reduction.MEAN)

        with tf.variable_scope('optimizer'):
            loss = tf.losses.softmax_cross_entropy(self.labels, logit, label_smoothing=0.1, reduction=tf.losses.Reduction.MEAN)

            l2_loss = self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
            )
            total_loss = loss + l2_loss + auxiliary_loss

            lossavg = tf.train.ExponentialMovingAverage(0.9, name='loss_moveavg')
            lossavg_op = lossavg.apply([loss, total_loss])
            with tf.control_dependencies([lossavg_op]):
                self.total_loss = tf.identity(total_loss)
            var_list = tf.trainable_variables()
            varavg = tf.train.ExponentialMovingAverage(0.9, name='var_moveavg')
            varavg_op = varavg.apply(var_list)
            optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.9, decay=0.9, epsilon=1.)
            grads = optimizer.compute_gradients(self.total_loss)
            clipped_grads = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in grads]
            train_op = optimizer.apply_gradients(clipped_grads, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([update_ops, lossavg_op, varavg_op, train_op])
            self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(self.logit, 1), tf.argmax(self.labels, 1)
                    ), tf.float32
                ), name='accuracy')

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

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, padding, activation=tf.nn.relu):
        assert(padding in ['same', 'valid'])
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            data_format=self.data_format,
            padding=padding
        )
        bn = tf.layers.batch_normalization(
            inputs=conv,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        if activation is not None:
            return activation(bn)
        else:
            return bn

    # squeeze-and-excitation block
    def squeeze_and_excitation(self, bottom):
        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        channels = bottom.get_shape()[1] if self.data_format == 'channels_first' else bottom.get_shape()[3]
        squeeze = tf.reduce_mean(
            input_tensor=bottom,
            axis=axes,
            keepdims=False,
        )
        excitation = tf.layers.dense(
            inputs=squeeze,
            units=int(channels // self.reduction),
            activation=tf.nn.relu,
        )
        excitation = tf.layers.dense(
            inputs=excitation,
            units=channels,
            activation=tf.nn.sigmoid,
        )
        weight = tf.reshape(excitation, [-1, 1, 1, channels])
        scaled = weight * bottom
        return scaled

    def _max_pooling(self, bottom, pool_size, strides, padding, name):
        assert(padding in ['same', 'valid'])
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, padding, name):
        assert(padding in ['same', 'valid'])
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
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

    def _inception_block1(self, bottom, filters, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_pool1x1'):
                branch_pool1x1 = self._avg_pooling(bottom, 3, 1, 'same', 'avg_pool')
                branch_pool1x1 = self._conv_bn_activation(branch_pool1x1, filters[3], 1, 1, 'same')
            with tf.variable_scope('branch_1x1'):
                branch_1x1 = self._conv_bn_activation(bottom, filters[0], 1, 1, 'same')
            with tf.variable_scope('branch_1x1x3x3'):
                branch_1x1x3x3 = self._conv_bn_activation(bottom, filters[1], 1, 1, 'same')
                branch_1x1x3x3 = self._conv_bn_activation(branch_1x1x3x3, filters[0], 3, 1, 'same')
            with tf.variable_scope('branch_1x1x5x5'):
                branch_1x1x5x5 = self._conv_bn_activation(bottom, filters[0], 1, 1, 'same')
                branch_1x1x5x5 = self._conv_bn_activation(branch_1x1x5x5, filters[2], 3, 1, 'same')
                branch_1x1x5x5 = self._conv_bn_activation(branch_1x1x5x5, filters[2], 3, 1, 'same')
            axes = 3 if self.data_format == 'channels_last' else 1
            if self.is_SENet:
                return self.squeeze_and_excitation(tf.concat([branch_1x1, branch_pool1x1, branch_1x1x3x3, branch_1x1x5x5], axis=axes))
            else:
                return tf.concat([branch_1x1, branch_pool1x1, branch_1x1x3x3, branch_1x1x5x5], axis=axes)

    def _grid_size_reduction1(self, bottom, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_3x3'):
                branch_3x3 = self._conv_bn_activation(bottom, 384, 3, 2, 'same')
            with tf.variable_scope('branch_1x1x5x5'):
                branch_1x1x5x5 = self._conv_bn_activation(bottom, 64, 1, 1, 'same')
                branch_1x1x5x5 = self._conv_bn_activation(branch_1x1x5x5, 96, 3, 1, 'same')
                branch_1x1x5x5 = self._conv_bn_activation(branch_1x1x5x5, 96, 3, 2, 'same')
            with tf.variable_scope('max_pool'):
                branch_pool = self._max_pooling(bottom, 3, 2, 'same', 'pool')
            axes = 3 if self.data_format == 'channels_last' else 1
            return tf.concat([branch_3x3, branch_1x1x5x5, branch_pool], axis=axes)

    def _inception_block2(self, bottom, filters, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_1x1'):
                branch_1x1 = self._conv_bn_activation(bottom, filters[0], 1, 1, 'same')
            with tf.variable_scope('branch_1x1x7x7'):
                branch_1x1x7x7 = self._conv_bn_activation(bottom, filters[1], 1, 1, 'same')
                branch_1x1x7x7 = self._conv_bn_activation(branch_1x1x7x7, filters[1], [1, 7], 1, 'same')
                branch_1x1x7x7 = self._conv_bn_activation(branch_1x1x7x7, filters[0], [7, 1], 1, 'same')
            with tf.variable_scope('branch_1x1x7x7x7x7'):
                branch_1x1x7x7x7x7 = self._conv_bn_activation(bottom, filters[1], 1, 1, 'same')
                branch_1x1x7x7x7x7 = self._conv_bn_activation(branch_1x1x7x7x7x7, filters[1], [7, 1], 1, 'same')
                branch_1x1x7x7x7x7 = self._conv_bn_activation(branch_1x1x7x7x7x7, filters[1], [1, 7], 1, 'same')
                branch_1x1x7x7x7x7 = self._conv_bn_activation(branch_1x1x7x7x7x7, filters[1], [7, 1], 1, 'same')
                branch_1x1x7x7x7x7 = self._conv_bn_activation(branch_1x1x7x7x7x7, filters[0], [1, 7], 1, 'same')
            with tf.variable_scope('branch_pool1x1'):
                branch_pool1x1 = self._avg_pooling(bottom, 3, 1, 'same', 'pool')
                branch_pool1x1 = self._conv_bn_activation(branch_pool1x1, filters[0], 1, 1, 'same')
            axes = 3 if self.data_format == 'channels_last' else 1
            if self.is_SENet:
                return self.squeeze_and_excitation( tf.concat([branch_1x1, branch_1x1x7x7, branch_1x1x7x7x7x7, branch_pool1x1], axis=axes))
            else:
                return tf.concat([branch_1x1, branch_1x1x7x7, branch_1x1x7x7x7x7, branch_pool1x1], axis=axes)

    def _grid_size_reduction2(self, bottom, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_1x1x3x3'):
                branch_1x1x3x3 = self._conv_bn_activation(bottom, 192, 1, 1, 'same')
                branch_1x1x3x3 = self._conv_bn_activation(branch_1x1x3x3, 320, 3, 2, 'same')
            with tf.variable_scope('branch_1x1x7x7x3x3'):
                branch_1x1x7x7x3x3 = self._conv_bn_activation(bottom, 192, 1, 1, 'same')
                branch_1x1x7x7x3x3 = self._conv_bn_activation(branch_1x1x7x7x3x3, 192, [1, 7], 1, 'same')
                branch_1x1x7x7x3x3 = self._conv_bn_activation(branch_1x1x7x7x3x3, 192, [7, 1], 1, 'same')
                branch_1x1x7x7x3x3 = self._conv_bn_activation(branch_1x1x7x7x3x3, 192, 3, 2, 'same')
            with tf.variable_scope('branch_pool'):
                branch_pool = self._max_pooling(bottom, 3, 2, 'same', 'pool')
            axes = 3 if self.data_format == 'channels_last' else 1
            return tf.concat([branch_1x1x3x3, branch_1x1x7x7x3x3, branch_pool], axis=axes)

    def _inception_block3(self, bottom, filters, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('branch_1x1'):
                branch_1x1 = self._conv_bn_activation(bottom, filters[0], 1, 1, 'same')
            with tf.variable_scope('branch_1x1x3x3'):
                branch_1x1x3x3 = self._conv_bn_activation(bottom, filters[1], 1, 1, 'same')
                branch_1x1x3x3x1x3 = self._conv_bn_activation(branch_1x1x3x3, filters[1], [1, 3], 1, 'same')
                branch_1x1x3x3x3x1 = self._conv_bn_activation(branch_1x1x3x3, filters[1], [3, 1], 1, 'same')
                axes = 3 if self.data_format == 'channels_last' else 1
                branch_1x1x3x3 = tf.concat([branch_1x1x3x3x1x3, branch_1x1x3x3x3x1], axis=axes)
            with tf.variable_scope('branch_1x1x3x3x3x3'):
                branch_1x1x3x3x3x3 = self._conv_bn_activation(bottom, filters[2], 1, 1, 'same')
                branch_1x1x3x3x3x3 = self._conv_bn_activation(branch_1x1x3x3x3x3, filters[1], 3, 1, 'same')
                branch_1x1x3x3x3x3x1x3 = self._conv_bn_activation(branch_1x1x3x3x3x3, filters[1], [1, 3], 1, 'same')
                branch_1x1x3x3x3x3x3x1 = self._conv_bn_activation(branch_1x1x3x3x3x3, filters[1], [3, 1], 1, 'same')
                branch_1x1x3x3x3x3 = tf.concat([branch_1x1x3x3x3x3x1x3, branch_1x1x3x3x3x3x3x1], axis=axes)
            with tf.variable_scope('branch_pool1x1'):
                branch_pool1x1 = self._avg_pooling(bottom, 3, 1, 'same', 'pool')
                branch_pool1x1 = self._conv_bn_activation(branch_pool1x1, filters[3], 1, 1, 'same')
            if self.is_SENet:
                return self.squeeze_and_excitation(tf.concat([branch_1x1, branch_1x1x3x3, branch_1x1x3x3x3x3, branch_pool1x1], axis=axes))
            else:
                return tf.concat([branch_1x1, branch_1x1x3x3, branch_1x1x3x3x3x3, branch_pool1x1], axis=axes)

    def _compute_output_shape(self, shape, kernel, padding, strides):
        assert(padding in ['same', 'valid'])
        if padding == 'valid':
            padded = 0
        else:
            padded = kernel // 2
        return (shape - kernel + 2 * padded) // strides  + 1
