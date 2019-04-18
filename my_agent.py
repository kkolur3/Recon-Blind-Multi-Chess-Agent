#!/usr/bin/env python3

"""
File Name:      my_agent.py
Authors:        Keshav Kolur and Anuj Bhyravabhotla
Date:           TODO: The date you finally started working on this.

Description:    Python file for my agent.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
from player import Player
import numpy as np


# TODO: Rename this class to what you would like your bot to be named during the game.
class MusicalChairs(Player):

    def __init__(self):
        pass
        
    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        # TODO: implement this method
        self.board = board
        self.state = StateEncoding(color)
        self.color = color
        if color == chess.WHITE:
            self.board.set_fen("8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        else:
            self.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/8/8 w KQkq - 0 1")
        # initialize trained model
        # initialize trained classifier
        
    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        assert isinstance(self.board, chess.Board)
        if captured_piece:
            captured_piece = self.board.remove_piece_at(captured_square).piece_type
        self.board.turn = self.color
        self.state.update_state_after_opponent_move(captured_piece, captured_square)
        pass

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        # TODO: update this method
        assert isinstance(possible_sense, list)
        i = 0
        while i < possible_sense.__len__():
            sense = possible_sense[i]
            if sense < 8 or sense > 55 or sense % 8 == 0 or (sense + 1) % 8 == 0:
                possible_sense.remove(sense)
            else:
                i += 1
        return random.choice(possible_sense)

    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        # TODO: implement this method
        # Hint: until this method is implemented, any senses you make will be lost.
        assert isinstance(self.board, chess.Board)
        for sense in sense_result:
            self.board.set_piece_at(sense[0], sense[1])
        self.state.update_state_after_sense(sense_result)


    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move
        
        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)
        
        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """
        # TODO: update this method
        assert isinstance(self.board, chess.Board)
        choice = random.choice(possible_moves)
        while not self.board.is_legal(choice):
            choice = random.choice(possible_moves)
        return choice
        
    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        # TODO: implement this method
        assert isinstance(self.board, chess.Board)
        if captured_piece:
            self.board.remove_piece_at(captured_square)
        self.board.push(taken_move if taken_move is not None else chess.Move.null())
        print(self.board)
        print("==================================================================================")
        self.state.update_state_with_move(taken_move, captured_piece, captured_square)

        pass
        
    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        # TODO: implement this method
        pass


class StateEncoding():
    def __init__(self, color):
        self.color = color
        self.board = chess.Board()
        self.reward_map = {
            chess.PAWN: [
                0, 0, 0, 0, 0, 0, 0, 0,
                0.5, 1, 1, -2, -2, 1, 1, 0.5,
                0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5,
                0, 0, 0, 2, 2, 0, 0, 0,
                0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5,
                1, 1, 2, 3, 3, 2, 1, 1,
                5, 5, 5, 5, 5, 5, 5, 5,
                0, 0, 0, 0, 0, 0, 0, 0
            ],
            chess.KNIGHT: [
                -5, -4, -3, -3, -3, -3, -4, -5,
                -4, -2, 0, 0.5, 0.5, 0, -2, -4,
                -3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3,
                -3, 0, 1.5, 2.0, 2.0, 1.5, 0, -3,
                -3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3,
                -3, 0, 1, 1.5, 1.5, 1, 0, -3,
                -4, -2, 0, 0, 0, 0, -2, -4,
                -5, -4, -3, -3, -3, -3, -4, -5
            ],
            chess.BISHOP: [
                -2, -1, -1, -1, -1, -1, -1, -2,
                -1, 0.5, 0, 0, 0, 0, 0.5, -1,
                -1, 1, 1, 1, 1, 1, 1, -1,
                -1, 0, 1, 1, 1, 1, 0, -1,
                -1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1,
                -1, 0, 0.5, 1, 1, 0.5, 0, -1,
                -1, 0, 0, 0, 0, 0, 0, -1,
                -2, -1, -1, -1, -1, -1, -1, -2
            ],
            chess.ROOK: [
                0, 0, 0, 0.5, 0.5, 0, 0, 0,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                0.5, 1, 1, 1, 1, 1, 1, 0.5,
                0, 0, 0, 0, 0, 0, 0, 0
            ],
            chess.QUEEN: [
                -2, -1, -1, -0.5, -0.5, -1, -1, -2,
                -1, 0, 0.5, 0, 0, 0, 0, -1,
                -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1,
                0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
                -0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
                -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
                -1, 0, 0, 0, 0, 0, 0, -1,
                -2, -1, -1, -0.5, -0.5, -1, -1, -2
            ],
            chess.KING: [
                2, 3, 1, 0, 0, 1, 3, 2,
                2, 2, 0, 0, 0, 0, 2, 2,
                -1, -2, -2, -2, -2, -2, -2, -1,
                -2, -3, -3, -4, -4, -3, -3, -2,
                -3, -4, -4, -5, -5, -4, -4, -3,
                -3, -4, -4, -5, -5, -4, -4, -3,
                -3, -4, -4, -5, -5, -4, -4, -3,
                -3, -4, -4, -5, -5, -4, -4, -3,
            ]
        }
        if not color:
            for key in self.reward_map.keys():
                self.reward_map[key] = list(reversed(self.reward_map[key]))
        self.material_differential = 0
        self.dists = [
            [True, 0, 0, 0, 1, 0, 0], [True, 0, 1, 0, 0, 0, 0], [True, 0, 0, 1, 0, 0, 0], [True, 0, 0, 0, 0, 1, 0],
            [True, 0, 0, 0, 0, 0, 1], [True, 0, 0, 1, 0, 0, 0], [True, 0, 1, 0, 0, 0, 0], [True, 0, 0, 0, 1, 0, 0],
            [True, 1, 0, 0, 0, 0, 0], [True, 1, 0, 0, 0, 0, 0], [True, 1, 0, 0, 0, 0, 0], [True, 1, 0, 0, 0, 0, 0],
            [True, 1, 0, 0, 0, 0, 0], [True, 1, 0, 0, 0, 0, 0], [True, 1, 0, 0, 0, 0, 0], [True, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
            [False, 1, 0, 0, 0, 0, 0], [False, 1, 0, 0, 0, 0, 0], [False, 1, 0, 0, 0, 0, 0], [False, 1, 0, 0, 0, 0, 0],
            [False, 1, 0, 0, 0, 0, 0], [False, 1, 0, 0, 0, 0, 0], [False, 1, 0, 0, 0, 0, 0], [False, 1, 0, 0, 0, 0, 0],
            [False, 0, 0, 0, 1, 0, 0], [False, 0, 1, 0, 0, 0, 0], [False, 0, 0, 1, 0, 0, 0], [False, 0, 0, 0, 0, 1, 0],
            [False, 0, 0, 0, 0, 0, 1], [False, 0, 0, 1, 0, 0, 0], [False, 0, 1, 0, 0, 0, 0], [False, 0, 0, 0, 1, 0, 0],
        ]
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 1000
        }
        self.special_move_indices = {
            (2, 1, 0): 0,
            (1, 2, 0): 1,
            (-1, 2, 0): 2,
            (-2, 1, 0): 3,
            (-2, -1, 0): 4,
            (-1, -2, 0): 5,
            (-1, 2, 0): 6,
            (2, -1, 0): 7,
            (1, 0, 2): 8,
            (1, 0, 3): 9,
            (1, 0, 4): 10,
            (1, -1, 2): 11,
            (1, -1, 3): 12,
            (1, -1, 4): 13,
            (1, 1, 2): 14,
            (1, 1, 3): 15,
            (1, 1, 4): 16
        }
        self.unit_moves = {
            (1,0): 0,
            (1,1): 1,
            (0,1): 2,
            (-1,1): 3,
            (-1,0): 4,
            (-1,-1): 5,
            (0,-1): 6,
            (1,-1):7
        }

    def create_move_encoding(self, move):
        self.update_board()
        assert isinstance(move, chess.Move)
        index = 0
        move_key = (
            chess.square_rank(move.to_square) - chess.square_rank(move.from_square),
            chess.square_file(move.to_square) - chess.square_file(move.from_square),
            0 if move.promotion is None else move.promotion
        )
        init_vector = [
            0 for x in range(64*73)
        ]
        if move is None or move == chess.Move.null() or self.board.piece_at(move.from_square) is None:
            return init_vector
        if self.board.piece_at(move.from_square).piece_type == chess.KNIGHT \
                or move.promotion is not None and move.promotion != chess.QUEEN and move.promotion != 0:
            index = 56 + self.special_move_indices[move_key]
        else:
            king_steps = chess.square_distance(move.from_square, move.to_square)
            unit_vector = (move_key[0], move_key[1])
            if king_steps != 0:
                unit_vector = (move_key[0]/king_steps, move_key[1]/king_steps)

            if unit_vector != (0,0):
                index = 7 * self.unit_moves[unit_vector] + king_steps - 1

        print(index)
        init_vector[move.from_square * 73 + index] = 1
        return init_vector

    def create_move_from_encoding(self, encoding):
        square = 0
        move_type = 0
        found_move = False
        while square < 64:
            move_type = 0
            while move_type < 73:
                found_move = encoding[square * 73 + move_type] == 1
                if found_move:
                    break
                else:
                    move_type += 1
            if found_move:
                break
            else:
                square += 1
        from_square = square
        unit_vector = (0,0)
        promotion = None
        if move_type < 56:
            index = int(move_type/7)
            for key in self.unit_moves.keys():
                if self.unit_moves[key] == index:
                    unit_vector = (key[0] * (move_type + 1) % 7, key[1] * (move_type + 1) % 7)
                    break
        else:
            index = move_type - 56
            for key in self.special_move_indices.keys():
                if self.special_move_indices[key] == index:
                    unit_vector = key

        to_square = chess.square(chess.square_file(from_square) + unit_vector[1],
                                      chess.square_rank(from_square) + unit_vector[0])
        if chess.square_rank(to_square) == 7:
            if (unit_vector.__len__() == 3):
                promotion = unit_vector[2]

        return chess.Move(from_square, to_square, promotion=promotion)

    def believe_square_is_empty(self, square):
        self.dists[square] = [0,0,0,0,0,0,0]

    def is_empty(self, square):
        return self.dists[square] == [0, 0, 0, 0, 0, 0, 0]

    def update_board(self, threshold=0.8):
        print(self.board)
        self.board.clear()
        for square_index in range(64):
            square = self.dists[square_index]
            arr_len = square.__len__()
            color = False
            piece = None
            if not self.is_empty(square_index):
                color = square[0]
                piece = 0
                duplicates = False
                max_prob = -1
                for x in range(1, arr_len):
                    if max_prob < square[x]:
                        max_prob = square[x]
                        piece = x
                        duplicates = False
                    elif max_prob == square[x]:
                      duplicates = True
            if piece is None or duplicates or max_prob < threshold:
                self.board.remove_piece_at(square_index)
            else:
                self.board.set_piece_at(square_index, chess.Piece(piece, color))
        print("============  Before probability update  =============================================")
        print(self.board)
        self.board.turn = self.color
        print("============   After probability update  =============================================")

    def compute_reward(self):
        reward = self.material_differential
        self.update_board()
        for x in range(64):
            piece = self.board.piece_at(x)
            if piece is not None and piece.color == self.color:
                reward += self.reward_map[piece.piece_type][x]
        return reward - 1

    def compute_move_reward_change(self, move):
        assert isinstance(move, chess.Move)
        self.update_board()
        piece_moved = self.board.piece_at(move.from_square).piece_type
        if not self.board.is_legal(move):
            return -500
        else:
            return \
                self.reward_map[piece_moved][move.to_square] - self.reward_map[piece_moved][move.from_square]

    def update_state_with_move(self, move, captured_piece, captured_square):
        if move is not None:
            piece = self.board.piece_at(move.from_square).piece_type
            self.dists[move.from_square] = [0,0,0,0,0,0,0]
            move_vector = [self.color, 0, 0, 0, 0, 0, 0]
            if piece == chess.PAWN \
                    and (chess.square_rank(move.to_square) == 8 or chess.square_rank(move.to_square) == 1):
                piece = chess.QUEEN if move.promotion is None else move.promotion
            move_vector[piece] = 1
            self.dists[move.to_square] = move_vector
            if captured_piece:
                captured_piece = self.board.remove_piece_at(captured_square)
                if captured_piece is None:
                    self.material_differential += 1
                else:
                    self.material_differential += self.piece_values[captured_piece.piece_type]
        self.update_board()

    def drift(self):
        initial_vector = [not self.color, 0, 0, 0, 0, 0, 0]
        opp_state = StateEncoding(not self.color)
        opp_state.board = self.board
        opp_state.dists = self.dists
        opp_state.board.turn = opp_state.color
        opp_state.update_board(threshold=0.0)
        print(opp_state.board.legal_moves)
        for move in opp_state.board.legal_moves:
            assert isinstance(move, chess.Move)
            reward_diff = opp_state.compute_move_reward_change(move)
            end_square = move.to_square
            if not self.is_empty(end_square) and self.dists[end_square][0] == self.color:
                continue
            else:
                self.dists[end_square] = initial_vector
            print(reward_diff)
            prob_delta = (0.5 + (reward_diff/10)) / opp_state.board.legal_moves.count()
            piece = self.board.piece_at(move.from_square).piece_type
            self.dists[move.from_square][piece] -= prob_delta
            if self.dists[move.from_square][piece] < 0:
                self.dists[move.from_square][piece] = 0
            self.dists[move.to_square][piece] += prob_delta
            if self.dists[move.to_square][piece] > 1:
                self.dists[move.to_square][piece] = 1
        self.normalize_dist()

    def normalize_dist(self):
        normalization_factors = [0,0,0,0,0,0]
        for square in range(64):
            if not self.is_empty(square) and self.dists[square][0] != self.color:
                square_dist = self.dists[square]
                for piece in self.reward_map:
                    normalization_factors[piece-1] += square_dist[piece]
        for x in range(normalization_factors.__len__()):
            if normalization_factors[x] > 0:
                normalization_factors[x] = 4/normalization_factors[x]

        for square in range(64):
            if not self.is_empty(square) and self.dists[square][0] != self.color:
                square_dist = self.dists[square]
                for piece in self.reward_map:
                    square_dist[piece] *= normalization_factors[piece - 1]
        for square in range(64):
            for piece in range(1, 7):
                if self.dists[square][piece] == float('nan') or self.dists[square][piece] < 0:
                    self.dists[square][piece] = 0
                elif self.dists[square][piece] == float('inf') or self.dists[square][piece] > 1:
                    self.dists[square][piece] = 1
        print(self.dists)

    def update_state_after_opponent_move(self, capture, capture_square):
        # probability drift to uniform as the game progresses and the picture becomes cloudier
        if capture:
            piece = self.board.piece_at(capture_square).piece_type
            self.material_differential -= self.piece_values[piece]
            self.dists[capture_square] = [0, 0, 0, 0, 0, 0, 0]
        self.drift()

    def update_state_after_sense(self, observations):
        for observation in observations:
            square = observation[0]
            piece = observation[1]
            observation_vector = [not self.color, 0, 0, 0, 0, 0, 0]
            if piece is not None:
                observation_vector[0] = piece.color
                observation_vector[piece.piece_type] = 1
            else:
                observation_vector = [0,0,0,0,0,0,0]
            self.dists[square] = observation_vector
        self.update_board()

    def export(self):
        state_copy = np.copy(self.dists)
        for square in state_copy:
            if square.__len__() > 0:
                square[0] = 1 if square[0] == self.color else 0
        return state_copy

    def import_dist(self, dist):
        dist_copy = np.copy(dist)
        for square in dist_copy:
            square[0] = self.color if square[0] == 1 else not self.color
        self.dists = dist_copy






if __name__ == '__main__':
    state = StateEncoding(chess.WHITE)
    print(state.create_move_encoding(chess.Move.from_uci("e2e4")))
    print(state.create_move_encoding(chess.Move.from_uci("g1f3")))
    print(state.create_move_encoding(chess.Move.from_uci("b1c3")))
    print(state.create_move_encoding(chess.Move.from_uci("c1f4")))
    print(state.create_move_encoding(chess.Move.from_uci("a7a8q")))
    print(state.create_move_encoding(chess.Move.from_uci("a7a8b")))
    print(state.create_move_from_encoding(state.create_move_encoding(chess.Move.from_uci("e2e4"))))