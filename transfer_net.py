import tensorflow as tf


# 卷积层 upsampling
def conv2d(x, kernel_shape, strides):
    weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name='weight')
    conv = tf.nn.conv2d(x, weights, [1, strides, strides, 1], padding='SAME', name='conv')
    normalized = batch_norm(conv, kernel_shape[3])
    return normalized


# 反卷积  downsampling
def conv2d_transpose(x, input_filter, output_filter, kernel, strides):
    shape = [kernel, kernel, output_filter, input_filter]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    batch_size = tf.shape(x)[0]
    height = x.get_shape()[1] * strides
    width = x.get_shape()[2] * strides
    output_shape = tf.stack([batch_size, height, width, output_filter])
    deconv = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape,
                                    strides=[1, strides, strides, 1], padding='SAME', name='deconv')
    normalized = batch_norm(deconv, output_filter)
    return normalized


# 归一化卷积层 spatial batch normalization
def batch_norm(x, size):
    batch_mean, batch_variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, batch_mean, batch_variance, beta, scale, epsilon, name='batch_norm')


# 残差块 make it easy for the network to learn the identify function
def residual(x, filters, kernel, stride):
    conv1 = conv2d(x, [kernel, kernel, filters, filters], stride)
    conv2 = conv2d(tf.nn.relu(conv1), [kernel, kernel, filters, filters], stride)
    residual = x + conv2

    return residual


def net(image):
    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(conv2d(image, [9, 9, 3, 64], 1))

    with tf.variable_scope('res1'):
        res1 = residual(conv1, 64, 3, 1)

    with tf.variable_scope('res2'):
        res2 = residual(res1, 64, 3, 1)

    with tf.variable_scope('res3'):
        res3 = residual(res2, 64, 3, 1)

    with tf.variable_scope('res4'):
        res4 = residual(res3, 64, 3, 1)

    with tf.variable_scope('deconv1'):
        deconv1 = tf.nn.relu(conv2d_transpose(res4, 64, 64, 3, 2))

    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 64, 3, 2))

    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(conv2d_transpose(deconv2, 64, 3, 9, 1))

    y = (deconv3+1) * 127.5
    return y
