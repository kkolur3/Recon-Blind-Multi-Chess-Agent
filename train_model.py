import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os
import chess
import pickle
# from google.colab import files

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
# convFlat = tf.nn.tanh(tf.reshape(flattened, [1, train_length, rnn_size]))
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


def init_dist(color):
    #encoding = [color, pawn, knight, bishop, rook, queen, king]
    dist = np.zeros(shape=(64, 7))
    isWhite = (color == chess.WHITE)
    for i in range(64):
        if i == 0 or i == 7 or i == 56 or i == 63: # ROOK
            square_dist = [isWhite, 0, 0, 0, 1, 0, 0]
        elif i == 1 or i == 6 or i == 57 or i == 62: # KNIGHT
            square_dist = [isWhite, 0, 1, 0, 0, 0, 0]
        elif i == 2 or i == 5 or i == 58 or i == 61: # BISHOP
            square_dist = [isWhite, 0, 0, 1, 0, 0, 0]
        elif i == 3 or i == 59: # QUEEN
            square_dist = [isWhite, 0, 0, 0, 0, 1, 0]
        elif i == 4 or i == 60: # KING
            square_dist = [isWhite, 0, 0, 0, 0, 0, 1]
        elif (i > 7 and i < 16) or (i > 47 and i < 56):
            square_dist = [isWhite, 1, 0, 0, 0, 0, 0]
        else:
            square_dist = [0, 0, 0, 0, 0, 0, 0]
        if i > 47:
            square_dist[0] = square_dist[not isWhite]
        dist[i] = square_dist
    return dist

piece_to_idx = {" p ": 1, " n ": 2, " b ": 3, " r ": 4, " q ": 5, " k ": 6}
idx_to_piece = ["p", "n", "b", "r", "q", "k"]


def create_episodes():
    from my_agent import StateEncoding
    for color in [chess.WHITE, chess.BLACK]:
        state = StateEncoding(color)

        game_history_dir = os.getcwd() + "/GameHistory"
        if os.path.isfile(os.path.join(os.getcwd(), "/episodes.pkl")):
            with (open("episodes.pkl", "rb")) as f:
                episodes = pickle.load(f)
        else:
            episodes = dict()
        prevBoardDist = init_dist(color)
        action_tensor = np.zeros(shape=(1, 64*82)) # chess.move from uci method that u pass in move string
        hidden_stat = np.zeros((1, rnn_size))
        cell_stat = np.zeros((1, rnn_size))
        reward = 0.0

        turn = False
        senseTurn = False
        moveTurn = False
        boardState = ""
        boardDist = np.zeros(shape=(64, 7))
        senseLoc = None
        if color == chess.WHITE:
            colorString = "WHITE"
        else:
            colorString = "BLACK"
        next_action = np.zeros(shape=(1, 64*82))
        for filename in os.listdir(game_history_dir):
            # print(filename)
            lineList = None
            gameOverLine = None
            if color == chess.WHITE:
                filenamePlayer = filename
            else:
                filenamePlayer = filename[:-5] + "black.txt"
            if "game" in filename and filenamePlayer not in episodes.keys():
                f_id = open(os.path.join(game_history_dir, filename))
                lineList = f_id.readlines()
                if len(lineList) > 2:
                    gameOverLine = lineList[len(lineList) - 2]
            if lineList and gameOverLine:
                print("new game! " + filenamePlayer + " being added")
                if "game" in filename and "Game Over" in gameOverLine:
                    episode = []
                    print("Training on " + filenamePlayer)
                    for line in lineList:
                        if colorString in line:
                            turn = True
                        if turn and "Sense" in line:
                            senseTurn = True
                            senseLoc = line[-3:-1]
                            line_counter = 7
                        if turn:
                            boardState += (line + "\n")
                        lineSplit = line.split("|")
                        if turn and senseTurn and lineSplit[0].isdigit():
                            row = int(lineSplit[0])
                            for i in range(1, 9):
                                piece = lineSplit[i].lower()
                                if piece == " p " or piece == " n " or piece == " b " or piece == " r " or piece == " q " or piece == " k ":
                                    board_idx = (row - 1) * 8 + (i - 1)
                                    square_dist = np.zeros(7)
                                    if lineSplit[i].upper() == lineSplit[i] and color == chess.WHITE:
                                        square_dist[0] = 1
                                    elif lineSplit[i].lower() == lineSplit[i] and color == chess.BLACK:
                                        square_dist[0] = 1
                                    piece_dist = piece_to_idx[piece]
                                    square_dist[piece_dist] = 1
                                    boardDist[board_idx] = square_dist
                            line_counter -= 1
                            if line_counter < 0 and turn and senseTurn and moveTurn:
                                senseTurn = False
                                moveTurn = False
                                line_counter = 7

                        if turn and "Move" in line:
                            moveIdx = line.find("taken:") + 7
                            move = line[moveIdx:-1]
                            moveTurn = True
                            turn = False
                            # print("Sense Loc " + senseLoc)
                            # print("Move: " + move)
                            print(boardState)
                            # print(boardDist)
                            if move == "None":
                                move = "0000"
                            uciMove = chess.Move.from_uci(move)
                            action = state.create_move_encoding(uciMove)
                            state.update_state_with_move(uciMove, False, False)
                            reward = state.compute_reward()
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
                    if len(episode) > 0:
                        episodes[filenamePlayer] = episode
                    else:
                        os.remove(os.path.join(game_history_dir, filename))
    f = open("episodesCopy.pkl", "wb")
    pickle.dump(episodes, f)
    f.close()
    # return episodes


def train_network(iterations):
    saver = tf.train.Saver()
    saver.restore(sess, "model/prev_model.ckpt")
    with (open("episodes.pkl", "rb")) as f:
        episodes = pickle.load(f)
    print(type(episodes))
    for episode in episodes:
        print(episode)
        for i in range(iterations):
            lo_sum = np.array([0.0])

            prevBoards = []
            moves = []
            rewards = []
            curBoards = []
            hidden = []
            cell = []
            for turn in episodes[episode]:
                prevBoards.append(turn["prevBoard"])
                moves.append(turn["action"])
                rewards.append(turn["reward"])
                curBoards.append(turn["curBoard"])
                hidden.append(turn["hidden"])
                cell.append(turn["cell"])
                initial_hidden = hidden[0]
                initial_cell = cell[0]

            final_obs = curBoards[-1]
            final_move = moves[-1]
            final_reward = rewards[-1]
            final_hidden = hidden[-1]
            final_cell = hidden[-1]

            probs, hidden = forward_pass(np.array(final_obs).reshape((1, 64, 7, 1)),
                                         np.array(final_move).reshape((1, 64*82)), final_cell, final_hidden)
            q_value = final_reward + alpha * np.max(probs)
            num_turns = len(episodes[episode])

            lo, _ = sess.run([loss, train], feed_dict={board: np.array(curBoards).reshape((num_turns, 64, 7, 1)),
                                                              actions: np.array(moves),
                                                              hidden_state: initial_hidden,
                                                              cell_state: initial_cell,
                                                              q_val: np.array([q_value]),
                                                              train_length: num_turns})
            lo_sum += lo

    saver.save(sess, "model/cur_model.ckpt") # CHANGE TO EITHER PREV OR CUR
    # files.download("model/prev_model.ckpt.meta")
    print("Loss: %d" % lo_sum)

letter_to_row = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}


def convertLocToSqIdx(senseLoc):
    return (int(senseLoc[1]) - 1) * 8 + (letter_to_row[senseLoc[0]])


def make_move(state, possible_moves):
    boardDist = state.export().reshape(1, 64, 7, 1)
    saver = tf.train.Saver()

    saver.restore(sess, "model/cur_model.ckpt") # CHANGE TO EITHER PREV OR CUR

    max_q = -float("inf")
    best_move = None
    observedBoard = np.zeros((1, rnn_size))
    beliefBoard = np.zeros((1, rnn_size)) #state.dists
    for move in possible_moves:
        action = np.array(state.create_move_encoding(move)).reshape(1, 5248)
        probs, hidden = forward_pass(boardDist, action, observedBoard, beliefBoard)
        state_out = [hidden.c[:1, :], hidden.h[:1, :]]
        current_reward = state.compute_reward()
        action_reward = state.compute_move_reward_change(move)
        q_action = action_reward + current_reward + alpha * np.max(probs)
        if q_action > max_q:
            best_move = move
            max_q = q_action
        beliefBoard = hidden.c
        observedBoard = hidden.h
    return best_move
# # RUN THESE TO TRAIN NEW NETWORK, IF DESIRED, THEN COMMENT THESE LINES AND RUN TEST.PY

if __name__ == '__main__':
    create_episodes()
    train_network(100)

