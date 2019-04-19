import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os
import my_agent
import chess

rnn_size = 128
discount_factor = 0.9
alpha = 0.9
replay_memory_size = 1000
learning_rate = 0.001

in_width = 8
in_height = 8
in_channel = 7
num_outputs = 64 * 82

# Placeholders
sess = tf.InteractiveSession()
board = tf.placeholder(tf.float32, shape=(None, in_width * in_height, in_channel, 1), name="board")
actions = tf.placeholder(tf.float32, shape=(None, num_outputs), name="actions")
hidden_state = tf.placeholder(tf.float32, shape=(1, rnn_size), name="hidden")
cell_state = tf.placeholder(tf.float32, shape=(1, rnn_size), name="cell")
q_val = tf.placeholder(tf.float32, shape=1, name="q_val")
train_length = tf.placeholder(dtype=tf.int32)

# Create CNNs
conv_1_weights = tf.get_variable("conv_1_weights", shape=(3, 3, 1, 4),
                                 dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
conv_1_bias = tf.get_variable("conv_1_bias", shape=4, initializer=tf.constant_initializer(0.0))
conv_2_weights = tf.get_variable("conv_2_weights", shape=(3, 3, 4, 8),
                                 dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
conv_2_bias = tf.get_variable("conv_2_bias", shape=8, initializer=tf.constant_initializer(0.0))
conv_3_weights = tf.get_variable("conv_3_weights", shape=(3, 3, 8, 16),
                                 dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
conv_3_bias = tf.get_variable("conv_3_bias", shape=16, initializer=tf.constant_initializer(0.0))

# Run board through CNNs
conv1 = tf.nn.relu(tf.nn.conv2d(board, conv_1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv_1_bias)
pool1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
bn1 = tf.layers.batch_normalization(pool1)

conv2 = tf.nn.relu(tf.nn.conv2d(bn1, conv_2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv_2_bias)
pool2 = tf.nn.max_pool(conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
bn2 = tf.layers.batch_normalization(pool2)

conv3 = tf.nn.relu(tf.nn.conv2d(bn2, conv_3_weights, strides=[1, 1, 1, 1], padding="SAME") + conv_3_bias)
pool3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
bn3 = tf.layers.batch_normalization(pool3)

# Run action through fully connected layer
action_fc1_weights = tf.get_variable(name="action_fc1_weights", shape=[num_outputs, rnn_size], dtype="float32",
                             initializer=tf.contrib.layers.xavier_initializer())
action_fc1_bias = tf.get_variable(name="action_fc1_bias", shape=rnn_size, dtype='float32',
                                  initializer=tf.constant_initializer(0.0))
action_fc1 = tf.nn.tanh(tf.add(tf.matmul(actions, action_fc1_weights), action_fc1_bias))

# Combined board ran through CNN with action taken
reshaped = tf.reshape(bn3, shape=(-1, rnn_size))
concatenate = tf.concat([reshaped, action_fc1], axis=-1)

combined_action_board_weights = tf.get_variable(name="combined_action_board_weights", shape=(256, rnn_size),
                                                dtype="float32", initializer=tf.contrib.layers.xavier_initializer())
combined_action_board_bias = tf.get_variable(name="combined_action_bias", shape=rnn_size,
                                             dtype="float32", initializer=tf.constant_initializer(0.0))
action_board_fc2 = tf.add(tf.matmul(concatenate, combined_action_board_weights), combined_action_board_bias)

# Run through RNN
flattened = tf.contrib.slim.flatten(action_board_fc2)
convFlat = tf.nn.tanh(tf.reshape(flattened, [1, train_length, rnn_size]))
lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, convFlat, dtype="float32",
                                    initial_state=(rnn_cell.LSTMStateTuple(hidden_state, cell_state)))
# Output weights
output_weights = tf.get_variable(name="output_weights", shape=[rnn_size, num_outputs], dtype="float32",
                                 initializer=tf.contrib.layers.xavier_initializer())
output_bias = tf.get_variable(name="output_bias", shape=num_outputs,
                              initializer=tf.constant_initializer(0.0))
final_outputs = tf.reshape(outputs[0, -1], [1, rnn_size])
logits = tf.add(tf.matmul(final_outputs, output_weights), output_bias)

# Training
loss = tf.square(tf.reduce_max(logits) - q_val)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


def softmax(x):
    return 1.0 / (1 + np.exp(-x))


def forward_pass(input_tensor, action_tensor, hidden_state_tensor, cell_state_tensor):
    probs, hidden = sess.run([logits, states], feed_dict={board: input_tensor,
                                                          actions: action_tensor,
                                                          hidden_state: hidden_state_tensor,
                                                          cell_state: cell_state_tensor,
                                                          train_length: 1})
    return probs, hidden


def init_dist():
    #encoding = [color, pawn, knight, bishop, rook, queen, king]
    dist = np.zeros(shape=(64, 7))
    for i in range(64):
        if i == 0 or i == 7 or i == 56 or i == 63: # ROOK
            square_dist = [1, 0, 0, 0, 1, 0, 0]
        elif i == 1 or i == 6 or i == 57 or i == 62: # KNIGHT
            square_dist = [1, 0, 1, 0, 0, 0, 0]
        elif i == 2 or i == 5 or i == 58 or i == 61: # BISHOP
            square_dist = [1, 0, 0, 1, 0, 0, 0]
        elif i == 3 or i == 59: # QUEEN
            square_dist = [1, 0, 0, 0, 0, 1, 0]
        elif i == 4 or i == 60: # KING
            square_dist = [1, 0, 0, 0, 0, 0, 1]
        elif (i > 7 and i < 16) or (i > 47 and i < 56):
            square_dist = [1, 1, 0, 0, 0, 0, 0]
        else:
            square_dist = [0, 0, 0, 0, 0, 0, 0]
        if i > 47:
            square_dist[0] = -square_dist[0]
        dist[i] = square_dist
    return dist

piece_to_idx = {" p ": 1, " n ": 2, " b ": 3, " r ": 4, " q ": 5, " k ": 6}
idx_to_piece = ["p", "n", "b", "r", "q", "k"]


def create_episodes():
    state = my_agent.StateEncoding(chess.WHITE)

    game_history_dir = "/Users/keshav/Documents/CS 4649/Recon-Blind-Multi-Chess-Agent/GameHistory"
    episodes = []
    prevBoardDist = init_dist()
    curBoardDist = None
    action_tensor = np.zeros(shape=(1, 64*82)) # chess.move from uci method that u pass in move string
    hidden_stat = np.zeros((1, rnn_size))
    cell_stat = np.zeros((1, rnn_size))
    reward = 0.0
    # for filename in os.listdir(game_history_dir):
    #     if "game" in filename:
    filename = "/Users/keshav/Documents/CS 4649/Recon-Blind-Multi-Chess-Agent/GameHistory/2019-04-08_17-12-41-617253game_boards.txt"
    whiteTurn = False
    senseTurn = False
    moveTurn = False
    boardState = ""
    boardDist = np.zeros(shape=(64, 7))
    senseLoc = None
    move = None
    episode = []

    next_action = np.zeros(shape=(1, 64*82))
    f_id = open(filename)
    for line in f_id:
        if "WHITE" in line:
            whiteTurn = True
        if whiteTurn and "Sense" in line:
            senseTurn = True
            senseLoc = line[-3:-1]
            line_counter = 7
        if whiteTurn:
            boardState += (line + "\n")
        lineSplit = line.split("|")
        if whiteTurn and senseTurn and lineSplit[0].isdigit():
            row = int(lineSplit[0])
            for i in range(1, 9):
                piece = lineSplit[i].lower()
                if piece == " p " or piece == " n " or piece == " b " or piece == " r " or piece == " q " or piece == " k ":
                    board_idx = (row - 1) * 8 + (i - 1)
                    square_dist = np.zeros(7)
                    if lineSplit[i].upper() == lineSplit[i]:
                        square_dist[0] = 1
                    else:
                        square_dist[0] = -1
                    piece_dist = piece_to_idx[piece]
                    square_dist[piece_dist] = 1
                    boardDist[board_idx] = square_dist
            line_counter -= 1
            if line_counter < 0 and whiteTurn and senseTurn and moveTurn:
                senseTurn = False
                moveTurn = False
                line_counter = 7

        if whiteTurn and "Move" in line:
            move = line[-5:-1]
            moveTurn = True
            whiteTurn = False
            print("Sense Loc " + senseLoc)
            print("Move: " + move)
            print(boardState)
            # print(boardDist)
            if move == "None":
                move = "0000"
            uciMove = chess.Move.from_uci(move)
            action = state.create_move_encoding(uciMove)
            state.update_state_with_move(uciMove, False, False)
            probs, hidden = forward_pass(prevBoardDist.reshape((1, 64, 7, 1)),
                                         np.array(action_tensor), hidden_stat, cell_stat)
            episode.append({"prevBoard": prevBoardDist, "action": action, "reward": reward,
                             "curBoard": boardDist, "hidden": hidden.h, "cell": hidden.c})
            hidden_stat = hidden.h
            cell_stat = hidden.c
            action_tensor = next_action
            boardState = ""
            prevBoardDist = boardDist
            boardDist = np.zeros(shape=(64, 7))
    episodes.append(episode)
    return episodes


def train_network(iterations):
    episode = create_episodes()
    for i in range(iterations):
        lo_sum = np.array([0.0])
        prevBoards = []
        actions = []
        rewards = []
        curBoards = []
        hidden = []
        cell = []
        for instances in episode:
            prevBoards.append(instances["prevBoard"])
            actions.append(instances["action"])
            rewards.append(instances["reward"])
            curBoards.append(instances["curBoard"])
            hidden.append(instances["hidden"])
            cell.append(instances["cell"])
        initial_hidden = hidden[0]
        initial_cell = cell[0]

        final_obs = curBoards[-1]
        final_action = actions[-1]
        final_reward = rewards[-1]
        final_hidden = hidden[-1]
        final_cell = hidden[-1]

        probs, hidden = forward_pass(np.array(final_obs), np.array(final_action), final_cell, final_hidden)
        q_value = final_reward + alpha * np.max(probs)
        lo, _ = sess.run([loss, train], feed_dict={board: np.array(curBoards),
                                                          actions: np.array(actions),
                                                          hidden_state: np.zeros(shape=(1, rnn_size)),
                                                          cell_state: np.zeros(shape=(1, rnn_size)),
                                                          q_val: np.array([q_value]),
                                                          train_length: 10})
        lo_sum += lo
    print("Loss: " + lo_sum)


letter_to_row = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}


def convertLocToSqIdx(senseLoc):
    return (int(senseLoc[1]) - 1) * 8 + (letter_to_row[senseLoc[0]])

train_network(100)

