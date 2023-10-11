```"""
NIM GAME

Game instruction: https://www.youtube.com/watch?v=BMmynMEDQxY&feature=youtu.be

Two players take turns dividing the pile into two unequal piles.
A player loses when he cannot divide any pile into two unequal piles.

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

Preparing the environment:
    to install easyAI use command: "pip install easyAI"
"""

from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player


class NimGame(TwoPlayerGame):

    def __init__(self, players=None):
        """
        Game initialization

        Parameters:
            players (list): List of players playing the game.
            pile (list):  List containing the size of the pile that players will share during the game.
            possible_move_len (int): Number of possible moves that player can do on the pile.
            current_player (int): Number of the player who starts the game.
        """
        self.players = players
        self.pile = [10]
        self.possible_move_len = None
        self.current_player = 1

    def possible_moves(self):
        """
        Creates possible moves that player can do

        Returns:
            list: possible allowed moves
        """
        return [
            "%d,%d" % (i + 1, self.pile[i] - j)
            for i in range(len(self.pile))
            for j in range(1, self.pile[i]) if (self.pile[i] - j) != j
        ]

    def make_move(self, move):
        """
        The function responsible for processing the player's movement.
        The move (1,5) means subtracting 5 from the first pile.

        Parameters:
            move (list): The move that player did.
        """
        move = list(map(int, move.split(',')))
        position = move[0] - 1
        self.pile[position] -= move[1]
        self.pile.append(move[1])

    def win(self):
        """
        The function that checks whether one of the players has won.

        Returns:
            bool: Information whether one of the players has won.
        """
        return self.possible_move_len <= 0

    def is_over(self):
        """
        The function that stops game if any player has won.

        Returns:
            bool: Information whether game should be stopped.
        """
        return self.win()

    def scoring(self):
        """
        Gives a score to the current game (for the AI).
        Returns:
             int: AI obtain 100 points if win, 0 points when lose.
        """
        return 100 if self.win() else 0

    def show(self):
        """
        The function that display information about game.
        """
        self.possible_move_len = len(self.possible_moves())
        print(f" bones left in the pile {self.pile}")
        if self.possible_move_len <= 0 and self.current_player == 1:
            print(f"AI win")
        elif self.possible_move_len <= 0 and self.current_player == 2:
            print(f"Human win")


# Ai's planned number of moves forward.
ai = Negamax(1)

#Creating a game for two AI and human players.
game = NimGame([AI_Player(ai), Human_Player()])
#Start the game.
game.play()
```