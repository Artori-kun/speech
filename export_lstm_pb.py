import tensorflow as tf
from tensorflow.python.framework import graph_util
from constants import c

tf.compat.v1.disable_eager_execution()

model_folder ='/home/minhhieu/My Projects/speech/'

num_features = c.LSTM.FEATURES
num_hidden = c.LSTM.HIDDEN
batch_size=1
num_layers=1
num_classes=43

# Construct the graph. For detail comments, please see lstm_ctc.py
inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, None, num_features], name='InputData')
targets = tf.compat.v1.sparse_placeholder(tf.int32, name='LabelData')
seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='SeqLen')

cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden)
stack = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
outputs, _ = tf.compat.v1.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32, time_major =False)
shape = tf.shape(input=inputs)
batch_s, max_time_steps = shape[0], shape[1]
outputs = tf.reshape(outputs, [-1, num_hidden])
W = tf.Variable(tf.random.truncated_normal([num_hidden, num_classes], stddev=0.1))
b = tf.Variable(tf.constant(0., shape=[num_classes]))
logits = tf.matmul(outputs, W) + b
logits = tf.reshape(logits, [batch_s, -1, num_classes])
logits = tf.transpose(a=logits, perm=(1, 0, 2))
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
y = tf.compat.v1.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)

# Need output node name to convert checkpoint into protocol buffers.
output_node_names = "SparseToDense"

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    saver = tf.compat.v1.train.Saver()
    # Restore checkpoint (load weights and biases)
    saver.restore(sess, tf.train.latest_checkpoint(model_folder))

graph = tf.compat.v1.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.compat.v1.Session() as sess:
     tf.compat.v1.global_variables_initializer().run()
     saver = tf.compat.v1.train.Saver()
     saver.restore(sess, tf.train.latest_checkpoint(model_folder))

     # Output model's graph details for reference.
     tf.io.write_graph(sess.graph_def, model_folder, 'graph_lstm.txt', as_text=True)
     # Freeze the output graph.
     output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def, \
                                                            output_node_names.split(","))
     # Write it into .pb file.
     with tf.io.gfile.GFile("/home/minhhieu/My Projects/speech/model/lstm_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
