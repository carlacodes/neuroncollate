#import deepinsight
import sys
import tensorflow as tf
import os

with tf.device('/gpu:0'):       # Run nodes with GPU 0
    m1 = tf.constant([[3, 5]])
    m2 = tf.constant([[2],[4]])
    product = tf.matmul(m1, m2)

sess = tf.Session()
print(sess.run(product))

sess.close()