import random
import time
import numpy as np
import pickle


class TTTBoard(object):
    """
    Tic-tac-toe board (initialized empty).

    Board positions mapping:
    0|1|2
    -+-+-
    3|4|5
    -+-+-
    6|7|8
    the board positions map to a list: [0, 1, 2, 3, 4, 5, 6, 7, 8]

    Crosses are mapped to 1, circles are mapped to -1 and empty position are mapped to 0. p1 always play crosses.
    .|X|O
    -+-+-
    .|X|.
    -+-+-
    .|O|.
    this board is mapped to [0, 1, -1, 0, 1, 0, 0, -1, 0]

    """
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_board(self):
        return self.board

    def get_empty_pos(self):
        """Returns a list of indexes of the board's empty positions."""
        return [index for index, value in enumerate(self.get_board()) if value == 0]

    def set_move(self, player_order, move):
        """
        Apply player's move to the board.

        Parameters
        ------
        player_order: int
            1 for p1, -1 for p2.
        move: int
            Integer from 0-8 (inclusive) indicating player's move on the board.
        """
        self.board[move] = player_order

    def is_over(self):
        """
        Verifies if the game is over.

        Returns
        -------
        result: boolean
            True:  Game Over
            False: Not Over
        winner: int
            1:  player 1
            -1: player 2 / not over
            0:  tie
        """
        # init
        result, winner = False, -1
        
        # Check for winner logic
        board = np.array(self.get_board())
        for bool_board, player in [(board == -1, -1), (board == 1, 1)]:
            a, b, c, d, e, f, g, h, i = bool_board
            if e & (d & f | b & h | a & i | g & c) | a & (b & c | d & g) | i & (g & h | c & f):
                return True, player

        # Check for a tie
        if not self.get_empty_pos():
            result, winner = True, 0
            
        return result, winner

    def __str__(self):
        board = self.get_board()
        board = ["X" if x == 1 else x for x in board]
        board = ["O" if x == -1 else x for x in board]
        board = ["." if x == 0 else x for x in board]
        return "{b[0]} | {b[1]} | {b[2]}\n--+---+---\n{b[3]} | {b[4]} | {b[5]}\n--+---+---\n{b[6]} | {b[7]} | {b[8]}\n\n".format(b=board)


class TTTAgent(object):
    def __init__(self):
        pass

    def set_order(self, order):
        """Sets agent order of play: 1 for p1, -1 for p2."""
        self.order = order

    def get_order(self):
        return self.order

    def play_turn(self, board, print_board=True):
        """Returns a move choosen randomly."""
        empty_positions = board.get_empty_pos()
        move = random.choice(empty_positions)
        board.set_move(self.get_order(), move)
        if print_board:
            print(board)
        return move


class TTTHuman(TTTAgent):
    def __init__(self):
        pass

    def play_turn(self, board, print_board=True):
        """Asks user input for a move."""
        print("Player", self.get_order())
        valid_move = False
        while not valid_move:
            try:
                move = int(input("Enter move: "))
            except:
                move = None
            valid_move = move in board.get_empty_pos()
            if not valid_move:
                print("Move not valid!")
        board.set_move(self.get_order(), move)
        if print_board:
            print(board)
        return move


class TTTSmartAgent(TTTAgent):
    def __init__(self, model, random_move_prob=0):
        """
        Tic-tac-toe agent with a 2 layer NN mind model.

        Parameters
        ----------
        model: dict {"W1": [numpy Hx9 array], "W2": [numpy 9xH array]}
            Policy network model weights. H is the hidden layer size.
        random_move_prob: float [0 - 1]
            Probability of making a random move.
        """
        self.model = model
        self.random_move_prob = random_move_prob

    def get_model(self):
        return self.model

    def play_turn(self, board, print_board=True):
        """Stochastic sampled move according to NN mind model output."""
        model = self.get_model()
        board_input = np.asarray(board.get_board()).reshape(9, 1)
        # Feed board to agent's policy network
        _, _, _, out = feed_forward(model, board_input)
        try:
            out_norm = [float(x) / float(sum(out)) for x in out]
            move = int(np.random.choice(9, 1, p=out_norm)[0])
        except ZeroDivisionError:
            move = -1
        empty_positions = board.get_empty_pos()
        if move not in empty_positions or move == -1 or random.random() < self.random_move_prob:
            move = random.choice(empty_positions)
        board.set_move(self.get_order(), move)
        if print_board:
            print(board)
        return move


def process_training_data(player, winner, board_hist, move_hist):
    """
    Maps tic_tac_toe single game data to NN training examples and labels data.

    Parameters
    ----------
    player: int (1 or -1)
        model's owner player.
    winner: int
        1 if p1 won, -1 if p2 won, 0 if tie.
    board_hist: list of lists
        History of boards seen by the player prior its move.
    move_hist: list
        Moves taken to each of the boards in board_hist.

    Returns
    -------
    X: [numpy 9xN array]
        Training examples. N equals len(board_hist).
    y: [numpy 9xN array]
        Training labels.
    """

    X = np.array(board_hist).T

    if winner == player.get_order():
        moves = [[1 if i == m else 0 for i in range(9)] for m in move_hist]
    elif winner == 0:
        moves = [[1 if i == m else 0 for i in range(9)] for m in move_hist]
    else:
        moves = [[0 if i == m else 0 for i in range(9)] for m in move_hist]

    y = np.array(moves).T
    return X, y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """Derivative of the sigmoid activation function."""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (2.0 / (1.0 + np.exp(-2 * x))) - 1


def tanh_prime(x):
    """Derivative of the tanh activation function."""
    return 1 - tanh(x)**2


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    """Derivative of the ReLU activation function."""
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def feed_forward(model, X):
    """
    Feed_forward input X into a 2 layer NN model.

    Parameters
    ----------
    model: dict {"W1": [numpy Hx9 array], "W2": [numpy 9xH array]}
        NN weights. H is the hidden layer size.
    X: [numpy 9xN array]
        NN input. N is the number of training examples.

    Returns
    -------
    z2: [numpy HxN array]
        Hidden layer input.
    a2: [numpy HxN array]
        Hidden layer output.
    z3: [numpy 9xN array]
        Output layer input.
    a3: [numpy 9xN array]
        Output layer output.
    """
    z2 = model["W1"] @ X
    a2 = sigmoid(z2)
    z3 = model["W2"] @ a2
    a3 = sigmoid(z3)
    return z2, a2, z3, a3


def update_model(model, player, winner, board_hist, move_hist, learnig_rate):
    """
    Updates 2 layer policy network weights using gradient descent (policy gradients) according to the played game data.

    Parameters
    ----------
    model: dict {"W1": [numpy Hx9 array], "W2": [numpy 9xH array]}
        Policy network to be updated. H is the hidden layer size.
    player: int (1 or -1)
        model's owner player.
    winner: int
        1 if p1 won, -1 if p2 won, 0 if tie.
    board_hist: list of lists
        History of boards seen by the player prior its move.
    move_hist: list
        Moves taken to each of the boards in board_hist.
    learnig_rate: float
        Gradient descent learnig rate.

    Returns
    -------
        Updated policy network model.
    """

    X, y = process_training_data(player, winner, board_hist, move_hist)

    # Number of training examples
    _, n = y.shape

    z2, a2, z3, a3 = feed_forward(model, X)

    # Back-Propagation
    delta3 = a3 - y
    delta2 = model["W2"].T @ delta3 * sigmoid_prime(z2)

    # Policy gradients
    W1_grad = (1 / n) * delta2 @ X.T
    W2_grad = (1 / n) * delta3 @ a2.T

    # Gradient descent
    model["W1"] -= (learnig_rate / n) * W1_grad
    model["W2"] -= (learnig_rate / n) * W2_grad
    return model


def play(p1, p2, verbose=True, time_interval=.5):
    """
    Tic-tac-toe game simulator.

    Parameters
    ----------
    p1, p2: TTTAgent, TTTSmartAgent or TTTHuman.
        p1 play crosses, p2 play circles.
    verbose: boolean
        If True, print game result at the end of the game.
    time_interval: float
        Time interval between turns, in seconds.

    Returns
    -------
    winner: int (1, -1 or 0)
        Result of the game.
    """

    board = TTTBoard()
    p1.set_order(1)
    p2.set_order(-1)
    for i in range(5):
        p1.play_turn(board, verbose)
        if board.is_over()[0]:
            break
        time.sleep(time_interval)
        p2.play_turn(board, verbose)
        if board.is_over()[0]:
            break
        time.sleep(time_interval)
    winner = board.is_over()[1]
    if verbose:
        if winner == 0:
            print("It's a tie!")
        else:
            print("Player", winner, "won!")
    return winner


def play_agent(agent1, agent2):
    """
    Used to play against trained agents or watch them play against each other.

    Parameters
    ----------
    agent1: string
        Possible agents: SxS, SxD, human, random.
    agent2: string
        Possible agents: SxS, DxS, human, random.
    """
    try:
        if agent1 == "SxS":
            model1_SxS = pickle.load(open("model1_SxS.p", 'rb'))
            p1 = TTTSmartAgent(model1_SxS)
        elif agent1 == "SxD":
            model1_SxD = pickle.load(open("model1_SxD.p", 'rb'))
            p1 = TTTSmartAgent(model1_SxD)
        elif agent1 == "human":
            p1 = TTTHuman()
        elif agent1 == "random":
            p1 = TTTAgent()

        if agent2 == "SxS":
            model2_SxS = pickle.load(open("model2_SxS.p", 'rb'))
            p2 = TTTSmartAgent(model2_SxS)
        elif agent2 == "DxS":
            model2_DxS = pickle.load(open("model2_DxS.p", 'rb'))
            p2 = TTTSmartAgent(model2_DxS)
        elif agent2 == "human":
            p2 = TTTHuman()
        elif agent2 == "random":
            p2 = TTTAgent()
    except:
        print("Agents probably not trained. Make sure model names are like 'model1_SxS.p', 'model2_DxS.p', etc.")
    play(p1, p2)


def training_play(p1, p2):
    """
    Tic-tac-toe game simulator used for agent's training.

    Parameters
    ----------
    p1, p2: TTTSmartAgent.
        p1 play crosses, p2 play circles.

    Returns
    -------
    winner: int (1, -1 or 0)
        Result of the game.
    p1_board_hist, p2_board_hist: list of lists
        History of boards seen by the players prior theis moves.
    p1_move_hist, p2_move_hist: list
        Moves taken to each of the boards in pX_board_hist.
    """

    board = TTTBoard()
    p1.set_order(1)
    p2.set_order(-1)
    p1_board_hist = []
    p2_board_hist = []
    p1_move_hist = []
    p2_move_hist = []
    for i in range(5):
        p1_board_hist.append(board.get_board().copy())
        p1_move = p1.play_turn(board, print_board=False)
        p1_move_hist.append(p1_move)
        if board.is_over()[0]:
            break
        p2_board_hist.append(board.get_board().copy())
        p2_move = p2.play_turn(board, print_board=False)
        p2_move_hist.append(p2_move)
        if board.is_over()[0]:
            break
    winner = board.is_over()[1]
    return winner, p1_board_hist, p1_move_hist, p2_board_hist, p2_move_hist


def train(training_games, learnig_rate, random_move_prob, resume=False):
    """
    Train 4 TTTSmartAgent models.

    model1_SxS.p: p1 trained agains a p2 TTTSmartAgent training at the same time.
    model2_SxS.p: p2 trained agains a p1 TTTSmartAgent training at the same time.
    model1_SxD.p: p1 trained agains a dumb p2 TTTAgent (random player).
    model2_DxS.p: p2 trained agains a dumb p1 TTTAgent (random player).

    After training, dumps models to binary files using pickle.

    Parameters
    ----------
    training_games: int
        Number of training games. Policy network is updated after each game.
    learnig_rate: float
        Learning rate used to update the policy network.
    resume: boolean
        If true, load pre-trained models, and resume trainig from them using the new parameters.
    """

    if resume:
        model1_SxS = pickle.load(open("model1_SxS.p", 'rb'))
        model2_SxS = pickle.load(open("model2_SxS.p", 'rb'))
        model1_SxD = pickle.load(open("model1_SxD.p", 'rb'))
        model2_DxS = pickle.load(open("model2_DxS.p", 'rb'))
    else:
        # Hidden layer size
        H = 1000
        # Inicialize models
        model1_SxS = {}
        model1_SxS["W1"] = np.random.rand(H, 9) * 2 - 1
        model1_SxS["W2"] = np.random.rand(9, H) * 2 - 1
        model2_SxS = {}
        model2_SxS["W1"] = np.random.rand(H, 9) * 2 - 1
        model2_SxS["W2"] = np.random.rand(9, H) * 2 - 1
        model1_SxD = {}
        model1_SxD["W1"] = np.random.rand(H, 9) * 2 - 1
        model1_SxD["W2"] = np.random.rand(9, H) * 2 - 1
        model2_DxS = {}
        model2_DxS["W1"] = np.random.rand(H, 9) * 2 - 1
        model2_DxS["W2"] = np.random.rand(9, H) * 2 - 1

    p2_SxD = TTTAgent()
    p1_DxS = TTTAgent()
    for i in range(training_games):
        p1_SxS = TTTSmartAgent(model1_SxS, random_move_prob)
        p2_SxS = TTTSmartAgent(model2_SxS, random_move_prob)
        p1_SxD = TTTSmartAgent(model1_SxD, random_move_prob)
        p2_DxS = TTTSmartAgent(model2_DxS, random_move_prob)

        winner, p1_board_hist, p1_move_hist, p2_board_hist, p2_move_hist = training_play(p1_SxS, p2_SxS)
        model1_SxS = update_model(model1_SxS, p1_SxS, winner, p1_board_hist, p1_move_hist, learnig_rate)
        model2_SxS = update_model(model2_SxS, p2_SxS, winner, p2_board_hist, p2_move_hist, learnig_rate)

        winner, p1_board_hist, p1_move_hist, _, _ = training_play(p1_SxD, p2_SxD)
        model1_SxD = update_model(model1_SxD, p1_SxD, winner, p1_board_hist, p1_move_hist, learnig_rate)

        winner, _, _, p2_board_hist, p2_move_hist = training_play(p1_DxS, p2_DxS)
        model2_DxS = update_model(model2_DxS, p2_DxS, winner, p2_board_hist, p2_move_hist, learnig_rate)

        if i % (training_games / 100) == 0:
            print(i * 100 / training_games, "%")

    pickle.dump(model1_SxS, open("model1_SxS.p", 'wb'))
    pickle.dump(model2_SxS, open("model2_SxS.p", 'wb'))
    pickle.dump(model1_SxD, open("model1_SxD.p", 'wb'))
    pickle.dump(model2_DxS, open("model2_DxS.p", 'wb'))


def test_agents(TRIALS):
    """
    Test agents in 7 different settings.

    TTTSmartAgent(model1_SxS) vs TTTSmartAgent(model2_SxS)
    TTTSmartAgent(model1_SxD) vs TTTSmartAgent(model2_DxS)
    TTTSmartAgent(model1_SxS) vs TTTAgent()
    TTTSmartAgent(model1_SxD) vs TTTAgent()
    TTTAgent()                vs TTTSmartAgent(model2_SxS)
    TTTAgent()                vs TTTSmartAgent(model2_DxS)
    TTTAgent()                vs TTTAgent()

    Parameters
    ----------
    TRIALS: int
        Number of games to be played at each setting.
    """

    model1_SxS = pickle.load(open("model1_SxS.p", "rb"))
    model2_SxS = pickle.load(open("model2_SxS.p", "rb"))
    model1_SxD = pickle.load(open("model1_SxD.p", "rb"))
    model2_DxS = pickle.load(open("model2_DxS.p", "rb"))

    smart1_SxS = TTTSmartAgent(model1_SxS)
    smart2_SxS = TTTSmartAgent(model2_SxS)
    smart1_SxD = TTTSmartAgent(model1_SxD)
    smart2_DxS = TTTSmartAgent(model2_DxS)
    dumb1 = TTTAgent()
    dumb2 = TTTAgent()

    # smart1_SxS vs smart2_SxS
    hist = []
    for i in range(TRIALS):
        winner = play(smart1_SxS, smart2_SxS, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTSmartAgent(model1_SxS) wins:", hist.count(1) / TRIALS)
    print("TTTSmartAgent(model2_SxS) wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")

    # smart1_SxD vs smart2_SxD
    hist = []
    for i in range(TRIALS):
        winner = play(smart1_SxD, smart2_DxS, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTSmartAgent(model1_SxD) wins:", hist.count(1) / TRIALS)
    print("TTTSmartAgent(model2_DxS) wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")

    # smart1_SxS vs dumb2
    hist = []
    for i in range(TRIALS):
        winner = play(smart1_SxS, dumb2, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTSmartAgent(model1_SxS) wins:", hist.count(1) / TRIALS)
    print("TTTAgent() wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")

    # smart1_SxD vs dumb2
    hist = []
    for i in range(TRIALS):
        winner = play(smart1_SxD, dumb2, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTSmartAgent(model1_SxD) wins:", hist.count(1) / TRIALS)
    print("TTTAgent() wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")

    # dumb1 vs smart2_SxS
    hist = []
    for i in range(TRIALS):
        winner = play(dumb1, smart2_SxS, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTAgent() wins:", hist.count(1) / TRIALS)
    print("TTTSmartAgent(model2_SxS) wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")

    # dumb1 vs smart2_DxS
    hist = []
    for i in range(TRIALS):
        winner = play(dumb1, smart2_DxS, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTAgent() wins:", hist.count(1) / TRIALS)
    print("TTTSmartAgent(model2_DxS) wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")

    # test4: dumb1 vs dumb2
    hist = []
    for i in range(TRIALS):
        winner = play(dumb1, dumb2, verbose=False, time_interval=0)
        hist.append(winner)
    print("TTTAgent() wins:", hist.count(1) / TRIALS)
    print("TTTAgent() wins:", hist.count(-1) / TRIALS)
    print("Ties:", hist.count(0) / TRIALS)
    print("\n")
