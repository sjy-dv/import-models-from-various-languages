import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, 'model.ckpt')

input_tensor = graph.get_tensor_by_name('input:0')
output_tensor = graph.get_tensor_by_name('output:0')

import numpy as np
input_data = np.array([[1.0, 2.0]])

with tf.Session(graph=graph) as sess:
    output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})

print(output_data)
