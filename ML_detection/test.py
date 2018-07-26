import numpy as np
import time
import numba
import tensorflow as tf

A = tf.random_normal([10,10])
B = tf.random_normal([10,10])


start_time = time.time()
C = np.dot(A,B)

#print ("time elapsed: {:.2f}s".format(time.time() - start_time))





start_time = time.time()
C = tf.matmul(A,B)
with tf.Session() as sess:
    result = sess.run(C)

#print ("time elapsed: {:.2f}s".format(time.time() - start_time))
#print(result)
mat = np.zeros(shape=(5, 1),dtype=complex)

a = np.arange(3)-2

b = np.linalg.norm(a) * np.linalg.norm(a)
print(b)
#print(np.matrix([0j] * 5).T)
