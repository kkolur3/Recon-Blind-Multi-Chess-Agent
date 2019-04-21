import my_agent
import random_agent
import play_game
import time

if __name__ == '__main__':
    wins = 0
    losses = 0
    for x in range(100):
        print("Playing new game")
        winning_color, winning_reason = play_game.play_local_game(my_agent.KnightFall(), random_agent.Random(),
                                                                  ["me", "the other guy"])
        # winning_color, winning_reason = play_game.play_local_game(random_agent.Random(), random_agent.Random(),
        #                                                           ["me", "the other guy"])

        print(winning_reason)
        if winning_color:
            wins += 1
        else:
            losses += 1

        print(str.format("Current record, {} - {}", wins, losses))
        print(str.format("Winning percentage: {}%", (1.0 * wins/(wins + losses)) * 100))
        # time.sleep(5)

    print(str.format("Final record, {} - {}", wins, losses))
    print(str.format("Winning percentage: {}%", (1.0 * wins / (wins + losses)) * 100))