import my_agent
import random_agent
import play_game
import time
import random
import chess

if __name__ == '__main__':
    winasBlack = 0
    winasWhite = 0
    lossasBlack = 0
    lossasWhite = 0
    totalWins = 0
    totalLosses = 0
    for x in range(100):

        if random.random() >= 0.5:
            agentColor = chess.WHITE
            winning_color, winning_reason = play_game.play_local_game(my_agent.KnightFall(), random_agent.Random(),
                                                                      ["me", "the other guy"])
        else:
            agentColor = chess.BLACK
            winning_color, winning_reason = play_game.play_local_game(random_agent.Random(), my_agent.KnightFall(),
                                                                      ["the other guy", "me"])

        # winning_color, winning_reason = play_game.play_local_game(my_agent.KnightFall(), my_agent.KnightFall(),
        #                                                           ["me", "the other guy"])
        # if winning_color == chess.WHITE:
        #     winasWhite += 1
        # else:
        #     winasBlack += 1
        # print(str.format("WHITE wins {}, BLACK wins {}", winasWhite, winasBlack))

        if agentColor == chess.WHITE:
            colorString = "WHITE"
        else:
            colorString = "BLACK"
        print("Played new game as " + colorString)
        print(winning_reason)
        if winning_color == agentColor:
            if agentColor == chess.WHITE:
                winasWhite += 1
            else:
                winasBlack += 1
        else:
            if agentColor == chess.WHITE:
                lossasWhite += 1
            else:
                lossasBlack += 1
        totalWins = winasWhite + winasBlack
        totalLosses = lossasWhite + lossasBlack
        print(str.format("Current Total record, {} - {}", totalWins, totalLosses))
        print(str.format("Winning percentage: {}%", (1.0 * totalWins/(totalWins + totalLosses)) * 100))
        print(str.format("Current White record, {} - {}", winasWhite, lossasWhite))
        if (winasWhite + lossasWhite) > 0:
            print(str.format("Winning Percentage as WHITE: {}%", (1.0 * winasWhite/(winasWhite + lossasWhite)) * 100))
        print(str.format("Current Black record, {} - {}", winasBlack, lossasBlack))
        if (winasBlack + lossasBlack) > 0:
            print(str.format("Winning Percentage as BLACK: {}%", (1.0 * winasBlack / (winasBlack + lossasBlack)) * 100))
        # time.sleep(5)