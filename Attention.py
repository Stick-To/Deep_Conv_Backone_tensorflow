from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class Attention:
    def __init__(self, config, input_shape, num_classes, weight_decay, data_format):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        assert data_format in ['channels_last', 'channels_first']
        self.data_format = data_format
        self.config = config
        assert len(config['ptr']) == 3
        assert config['ptr'][0] > 0 and config['ptr'][1] > 0 and config['ptr'][2] > 0
        assert len(config['attention_module_list']) == 3
        if data_format == 'channels_last':
            assert input_shape[0] // config['init_pooling_strides']*(2**config['first_attention_downsampling_times']) != 0 and \
                   input_shape[1] // config['init_pooling_strides']*(2**config['first_attention_downsampling_times']) != 0
        else:
            assert input_shape[1] // config['init_pooling_strides']*(2**config['first_attention_downsampling_times']) != 0 and \
                   input_shape[2] // config['init_pooling_strides']*(2**config['first_attention_downsampling_times']) != 0
        self.downsampling_times = [config['first_attention_downsampling_times'] - i for i in range(3)]
        self.p, self.t, self.r = config['ptr']
        self.attention_list = config['attention_module_list']
        self.filters_list = [config['init_conv_filters']*(2**i) for i in range(4)]

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
        with tf.variable_scope('init_conv'):
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
        residual_block1 = self._residual_bottleneck(pool1, self.filters_list[0], 1, 'res_1')
        attention_module1 = residual_block1
        for i in range(self.attention_list[0]):
            attention_module1 = self._attention_module(attention_module1, self.filters_list[0], self.downsampling_times[0], 'attention_module_1_'+str(i+1))
        residual_block2 = self._residual_bottleneck(attention_module1, self.filters_list[1], 2, 'res_2')
        attention_module2 = residual_block2
        for i in range(self.attention_list[1]):
            attention_module2 = self._attention_module(attention_module2, self.filters_list[1], self.downsampling_times[1], 'attention_module_2_'+str(i+1))
        residual_block3 = self._residual_bottleneck(attention_module2, self.filters_list[2], 2, 'res_3')
        attention_module3 = residual_block3
        for i in range(self.attention_list[2]):
            attention_module3 = self._attention_module(attention_module3, self.filters_list[2], self.downsampling_times[2], 'attention_module_3_'+str(i+1))
        residual_block4_1 = self._residual_bottleneck(attention_module3, self.filters_list[3], 2, 'res_4_1')
        residual_block4_2 = self._residual_bottleneck(residual_block4_1, self.filters_list[3], 1, 'res_4_2')
        residual_block4_3 = self._residual_bottleneck(residual_block4_2, self.filters_list[3], 1, 'res_4_3')
        bn = self._bn(residual_block4_3)
        relu = tf.nn.relu(bn)
        with tf.variable_scope('final_dense'):
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(relu, axis=axes, keepdims=False, name='global_pool')
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
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True)
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
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
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
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        )
        return conv

    def _residual_bottleneck(self, bottom, filters, strides, scope):
        with tf.variable_scope(scope):
            conv = self._bn_activation_conv(bottom, filters, 1, 1)
            conv = self._bn_activation_conv(conv, filters, 3, strides)
            conv = self._bn_activation_conv(conv, filters*4, 1, 1)
            shutcut = self._bn_activation_conv(bottom, filters*4, 1, strides)
            return conv + shutcut

    def _softmask(self, bottom, filters, downsampling_times, scope):
        with tf.variable_scope(scope):
            downsampling_layers = []
            residual_unit = self._max_pooling(bottom, 2, 2, 'down_1')
            for i in range(self.r):
                residual_unit = self._residual_bottleneck(residual_unit, filters, 1, 'down_res_1_'+str(i+1))
            for i in range(downsampling_times-1):
                downsampling_layers.append(residual_unit)
                downsampling = self._max_pooling(residual_unit, 2, 2, 'down_'+str(i+2))
                for j in range(self.r):
                    residual_unit = self._residual_bottleneck(downsampling, filters, 1, 'down_res_'+str(i+2)+'_'+str(j+1))
            for i in range(1, downsampling_times):
                if self.data_format == 'channels_first':
                    residual_unit = tf.transpose(residual_unit, [0, 2, 3, 1])
                    upsampling = tf.image.resize_bilinear(residual_unit, 2*[residual_unit.get_shape()[1], 2*residual_unit.get_shape()[2]], name='up_'+str(i))
                    upsampling = tf.transpose(upsampling, [0, 3, 1, 2])
                    residual_unit = upsampling + self._residual_bottleneck(downsampling_layers[-i], filters, 1, 'res_res_'+str(i))
                    for j in range(self.r):
                        residual_unit = self._residual_bottleneck(residual_unit, filters, 1, 'up_res_'+str(i)+'_'+str(j+1))
                else:
                    upsampling = tf.image.resize_bilinear(residual_unit, [2*residual_unit.get_shape()[1], 2*residual_unit.get_shape()[2]], name='up_'+str(i))
                    residual_unit = upsampling + self._residual_bottleneck(downsampling_layers[-i], filters, 1, 'res_res_'+str(i))
                    residual_unit = self._residual_bottleneck(residual_unit, filters, 1, 'up_res_'+str(i))
            if self.data_format == 'channels_first':
                residual_unit = tf.transpose(residual_unit, [0, 2, 3, 1])
                upsampling = tf.image.resize_bilinear(residual_unit, 2*[residual_unit.get_shape()[1], 2*residual_unit.get_shape()[2]], name='up_'+str(downsampling_times))
                upsampling = tf.transpose(upsampling, [0, 3, 1, 2])
            else:
                upsampling = tf.image.resize_bilinear(residual_unit, [2*residual_unit.get_shape()[1], 2*residual_unit.get_shape()[2]], name='up_'+str(downsampling_times))
            conv = self._conv_bn_activation(upsampling, filters*4, 1, 1, tf.nn.relu)
            conv = self._conv_bn_activation(conv, filters*4, 1, 1, tf.nn.sigmoid)
            return conv

    def _attention_module(self, bottom, filters, downsampling_times, scope):
        with tf.variable_scope(scope):
            residual_unit = bottom
            for i in range(self.p):
                residual_unit = self._residual_bottleneck(residual_unit, filters, 1, 'head_res_'+str(i))
            softmask = self._softmask(residual_unit, filters, downsampling_times, 'mask_branch')
            for i in range(self.t):
                residual_unit = self._residual_bottleneck(residual_unit, filters, 1, 'trunk_res_'+str(i))
            residual_attention = (1 + softmask) * residual_unit
            for i in range(self.p):
                residual_attention = self._residual_bottleneck(residual_attention, filters, 1, 'final_res_'+str(i))
            return residual_attention

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
