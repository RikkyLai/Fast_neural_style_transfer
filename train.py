import tensorflow as tf
import numpy as np
from scipy.misc import imread
from loss_net import content_layer_net, compute_total_loss
from transfer_net import net
from image_process import read_image, get_batches
import os
import cv2


TRAIN_IMAGE_PATH = 'train.list'
TEST_IMAGE_PATH = 'test/test.jpg'
LR_IMAGE_SIZE = 72
SR_IMAGE_SIZE = 288
learning_rate = 1e-3
BATCH_SIZE = 4

TRAIN_CHECK_POINT = 'model/trained_model/'
CHECK_POINT_PATH = 'log/'

EPOCH_NUM = 1
DATA_SIZE = 82783


def stylize_train(train_image_path):

    LR_input = tf.placeholder(dtype=tf.float32, shape=[None, LR_IMAGE_SIZE, LR_IMAGE_SIZE, 3], name='LR_input')
    SR_input = tf.placeholder(dtype=tf.float32, shape=[None, SR_IMAGE_SIZE, SR_IMAGE_SIZE, 3], name='SR_input')
    y_ = net(LR_input)
    content_net = content_layer_net(SR_input)

    content_cost, cost = compute_total_loss(content_net, y_)

    tf.summary.scalar('losses/content_loss', content_cost)
    tf.summary.scalar('losses/loss', cost)
    tf.summary.image('generated', y_)
    tf.summary.image('origin', LR_input)
    summary = tf.summary.merge_all()

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(CHECK_POINT_PATH, sess.graph)

        for epoch in range(EPOCH_NUM):
            batch_index = 0
            for i in range(DATA_SIZE//BATCH_SIZE):
                SR_image_batch, LR_image_batch, batch_index = get_batches(train_image_path, batch_index)
                _, content_loss, loss = sess.run([train_op, content_cost, cost], feed_dict={LR_input: LR_image_batch,
                                                                SR_input: SR_image_batch})
                step += 1

                if i % 10 == 0:
                    print('Epoch %d, Batch %d of %d, loss is %.3f, content loss is %.3f'%
                          (epoch + 1, i, DATA_SIZE // BATCH_SIZE, loss, content_loss))
                    # test_image = read_image(TEST_IMAGE_PATH)
                    # sr_image = sess.run(y_, feed_dict={LR_input: test_image})
                    # cv2.destroyAllWindows()
                    # image = np.clip(sr_image, 0, 255).astype(np.uint8)
                    # image = np.squeeze(image)
                    # cv2.imshow('result', image)
                    # cv2.waitKey(100)
                if i % 1000 == 0:
                    # save model parameters
                    saver.save(sess, os.path.join(TRAIN_CHECK_POINT, 'model.ckpt'), global_step=step)


stylize_train(TRAIN_IMAGE_PATH)
