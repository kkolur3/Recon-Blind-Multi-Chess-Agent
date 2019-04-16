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

#Placeholders
sess = tf.InteractiveSession()
board = tf.placeholder(tf.float32, shape=(in_width, in_height, in_channel), name="board")
actions = tf.placeholder(tf.float32, shape=(None, num_outputs), name="actions")
hidden_state = tf.placeholder(tf.float32, shape=(1, rnn_size), name="hidden")
cell_state = tf.placeholder(tf.float32, shape=(1, rnn_size), name="cell")
q_val = tf.placeholder(tf.float32, shape=1, name="q_val")
train_length=tf.placeholder(dtype=tf.int32)


