import tensorflow as tf
import numpy as np

rnn_size = 128
discount_factor = 0.9
alpha = 0.9
replay_memory_size = 1000
learning_rate = 0.001

in_width = 8
in_height = 8
in_channel = 7
num_outputs = 64 * 64 * 6

# Placeholders
sess = tf.InteractiveSession()
board = tf.placeholder(tf.float32, shape=(None, in_width, in_height, in_channel), name="board")
actions = tf.placeholder(tf.float32, shape=(None, num_outputs), name="actions")
hidden_state = tf.placeholder(tf.float32, shape=(1, rnn_size), name="hidden")
cell_state = tf.placeholder(tf.float32, shape=(1, rnn_size), name="cell")
q_val = tf.placeholder(tf.float32, shape=1, name="q_val")
train_length = tf.placeholder(dtype=tf.int32)

# Create CNNs
conv_1_weights = tf.get_variable("conv_1_weights", shape=(7, 7, 7, 32),
                                 dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
conv_1_bias = tf.get_variable("conv_1_bias", shape=32, initializer=tf.constant_initializer(0.0))
conv_2_weights = tf.get_variable("conv_2_weights", shape=(7, 7, 32, 64),
                                 dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
conv_2_bias = tf.get_variable("conv_2_bias", shape=64, initializer=tf.constant_initializer(0.0))
conv_3_weights = tf.get_variable("conv_3_weights", shape=(7, 7, 64, 128),
                                 dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
conv_3_bias = tf.get_variable("conv_3_bias", shape=128, initializer=tf.constant_initializer(0.0))

# Run input through CNNs
conv1 = tf.nn.relu(tf.nn.conv2d(board, conv_1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv_1_bias)
pool1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
bn1 = tf.layers.batch_normalization(pool1)

conv2 = tf.nn.relu(tf.nn.conv2d(bn1, conv_2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv_2_bias)
pool2 = tf.nn.max_pool(conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
bn2 = tf.layers.batch_normalization(pool2)

conv3 = tf.nn.relu(tf.nn.conv2d(bn2, conv_3_weights, strides=[1, 1, 1, 1], padding="SAME") + conv_3_bias)
pool3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
bn3 = tf.layers.batch_normalization(pool3)


