from __future__ import print_function

import time
import utils

import tensorflow as tf

from constants import c
from file_logger import FileLogger

# Parameters for index to string
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1

# decode dictionary
data_dir = 'data_record_web.lst'
dic = utils.load_encode_dic("encode_decode.json")


######
def make_cell(hidden_layer):
    return tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_layer, state_is_tuple=True)


# mfcc features
num_features = c.LSTM.FEATURES
# Accounting the 0th index +  space + blank label
# Processing Vietnamese speech require different character range

num_classes = dic["char_num"] + 1

# Hyper-parameters
num_hidden = c.LSTM.HIDDEN
batch_size = c.LSTM.BATCH_SIZE
num_epochs = 150
num_layers = 3

# Calculate ler every [num_steps] batch
num_steps = 20

# Directories for training, dev and log from conf.json
Dev_DIR = c.LSTM.DEV_PATH
Train_DIR = c.LSTM.TRAIN_PATH
Log_DIR = c.LSTM.LOG_PATH

total_train_sample = utils.get_metadata_len(Train_DIR)
total_dev_sample = utils.get_metadata_len(Dev_DIR)

# Validation list and val_batch_size
# total_dev_sample = utils.shuffle_every_epoch(total_dev_sample)
# dev_size = len(total_dev_sample)

# File log
file_logger_batch = FileLogger('out_batch.tsv', ['curr_epoch',
                                                 'batch',
                                                 'train_cost',
                                                 'train_ler',
                                                 'original',
                                                 'decode'])

file_logger_epoch = FileLogger('out_epoch.tsv', ['curr_epoch',
                                                 'train_cost',
                                                 'train_ler',
                                                 'val_cost',
                                                 'val_ler',
                                                 'val_original',
                                                 'val_decoded'])

graph = tf.Graph()
with graph.as_default():
    # Has size [batch_size, max_step_size, num_features], but the
    # batch_size and max_step_size can vary along each step
    inputs = tf.compat.v1.placeholder(tf.float32, [None, None, num_features], name='InputData')

    print(inputs.shape)

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.compat.v1.sparse_placeholder(tf.int32, name='LabelData')
    # 1d array of size [batch_size]
    seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='SeqLen')

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    # cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    # Stacking rnn cells
    stack = tf.compat.v1.nn.rnn_cell.MultiRNNCell([make_cell(num_hidden) for _ in range(num_layers)],
                                                  state_is_tuple=True)
    # The second output is the last state and we will no use that
    outputs, _ = tf.compat.v1.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32, time_major=False)
    # Inputs shape
    shape = tf.shape(input=inputs)
    # Get shape
    batch_s, max_time_steps = shape[0], shape[1]
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Weigts and biases
    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.random.truncated_normal([num_hidden, num_classes], stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))
    # Add dropout for W
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    W_drop = tf.nn.dropout(W, 1 - keep_prob)

    # Doing the affine projection
    logits = tf.matmul(outputs, W_drop) + b
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    # Time major
    logits = tf.transpose(a=logits, perm=(1, 0, 2))

    # ctc loss
    loss = tf.compat.v1.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(input_tensor=loss)

    # Gradient clipping
    tvars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(ys=cost, xs=tvars)
    grad_norm = tf.linalg.global_norm(grads, name='grads')
    grads, _ = tf.clip_by_global_norm(grads, 2, use_norm=grad_norm)
    grads = list(zip(grads, tvars))

    # Optimizer
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)

    # Op to update all variables according to their gradient
    optimizer = optimizer.apply_gradients(grads_and_vars=grads)

    # Decoded and label error rate
    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # Inaccuracy: label error rate
    ler = tf.reduce_mean(input_tensor=tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # Save model
    saver = tf.compat.v1.train.Saver()

    # Create a summary to monitor cost tensor
    summary_loss = tf.compat.v1.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    # summary_ler = tf.summary.scalar("ler", ler)

    # Create summaries to visualize weights
    # for var in tf.trainable_variables():
    #    tf.summary.histogram(var.name, var)
    # Summarize all gradients
    summary_grad = tf.compat.v1.summary.scalar("grad", grad_norm)
    # Merge all summaries into a single op
    merged_summary_op = tf.compat.v1.summary.merge_all()

# Start training
# train on one lst file only
with tf.compat.v1.Session(graph=graph) as sess:
    # Run the initializer
    # Initialize the variables (i.e. assign their default value)
    tf.compat.v1.global_variables_initializer().run()
    # Op to write logs to Tensorboard
    summary_writer = tf.compat.v1.summary.FileWriter(Log_DIR, graph=tf.compat.v1.get_default_graph())

    val_base = 200
    for curr_epoch in range(num_epochs):

        start = time.time()
        # Zero train cost & ler for each epoch
        train_cost = train_ler = 0
        val_cost = val_ler = 0
        # Shuffle training samples and get their npz path
        utils.shuffle_every_epoch(Train_DIR)
        # Total size of training samples
        # num_examples = total_train_sample
        #         print(train_list)
        #         print(num_examples)
        # Go through all samples for each epoch
        num_batches_per_epoch_train = int(total_train_sample / batch_size)
        num_batches_per_epoch_dev = int(total_dev_sample / batch_size)
        # Shuffle validation samples and get their npz path
        utils.shuffle_every_epoch(Dev_DIR)
        print("Done Shuffling")

        for batch in range(num_batches_per_epoch_train):
            # Get batch samples for training
            train_inputs, train_targets, train_seq_len, original = utils.next_batch_training(batch_size,
                                                                                             batch,
                                                                                             Train_DIR,
                                                                                             dic["encode"])
            # print(train_inputs.shape)
            # print(train_targets)
            # Feed_dict for training
            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len,
                    keep_prob: 0.8}
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary node
            _, batch_cost, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict=feed)

            # Collect batch cost
            train_cost += batch_cost
            # train_ler += sess.run(ler, feed_dict=feed)
            # Write logs at every iteration
            summary_writer.add_summary(summary, curr_epoch * num_batches_per_epoch_train + batch)

            if batch % num_steps == 0 and batch != 0:
                feed_ler = {inputs: train_inputs,
                            targets: train_targets,
                            seq_len: train_seq_len,
                            keep_prob: 1}
                # Calculate label error rate
                train_ler += sess.run(ler, feed_dict=feed_ler)
                # Decoding
                d = sess.run(decoded[0], feed_dict=feed_ler)
                d_last = tf.compat.v1.sparse_to_dense(d[0], d[2], d[1]).eval()

                str_decoded = utils.decode(d_last[-1], dic)

                # Log batch
                file_logger_batch.write([curr_epoch + 1,
                                         batch + 1,
                                         train_cost / batch,
                                         train_ler / (batch / num_steps),
                                         original[-1],
                                         str_decoded])
            # print("Done training a batch")

        # Train cost and ler for each epoch
        train_cost /= num_batches_per_epoch_train
        train_ler /= (num_batches_per_epoch_train / num_steps)
        # Validation
        for batch in range(num_batches_per_epoch_dev):

            val_inputs, val_targets, val_seq_len, val_original = utils.next_batch_training(total_dev_sample,
                                                                                           0,
                                                                                           Dev_DIR,
                                                                                           dic["encode"])
            # print(val_seq_len.shape)
            # print(val_inputs.shape)
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len,
                        keep_prob: 1.0}

            batch_cost, batch_ler = sess.run([cost, ler], feed_dict=val_feed)
            val_cost += batch_cost
            val_ler += val_ler

        val_cost /= num_batches_per_epoch_dev
        val_ler /= num_batches_per_epoch_dev

        # Save checkpoint when the val_cost reduces.
        if val_cost < val_base:
            saver.save(sess, './checkpoints/lstm_model', global_step=curr_epoch)
            val_base = val_cost

        # Decoding
        d = sess.run(decoded[0], feed_dict=val_feed)
        # Only recover the last sample in validation set.
        d_last = tf.compat.v1.sparse_to_dense(d[0], d[2], d[1]).eval()

        str_decoded = utils.decode(d_last[-1], dic)

        # Log epoch
        file_logger_epoch.write([curr_epoch + 1,
                                 train_cost,
                                 train_ler,
                                 val_cost,
                                 val_ler,
                                 val_original[-1],
                                 str_decoded])

        # file_writer = tf.summary.SummaryWriter('/home/minhhiu/My Projects/speech/logs', sess.graph)

        print('Original val: %s' % val_original[-1])
        print('Decoded val: %s' % str_decoded)
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
              "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))
        print(' ')
