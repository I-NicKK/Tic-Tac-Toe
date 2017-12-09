#!/usr/bin/env python3
import tic_tac_toe as ttt
import time
import argparse

parser = argparse.ArgumentParser(
    description="""
    Train 4 TTTSmartAgent models.

    model1_SxS.p: p1 trained agains a p2 TTTSmartAgent training at the same time.
    model2_SxS.p: p2 trained agains a p1 TTTSmartAgent training at the same time.
    model1_SxD.p: p1 trained agains a dumb p2 TTTAgent (random player).
    model2_DxS.p: p2 trained agains a dumb p1 TTTAgent (random player).

    After training, dumps models to binary files using pickle.
    """, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('training_games', type=int, help="Number of training games (policy network is updated after each game).")
parser.add_argument('learnig_rate', type=float, help='Learning rate used to update the policy network.')
parser.add_argument('random_move_prob', type=float, nargs="?", default=0, help='Probability of making a random move.')
parser.add_argument('resume', type=bool, nargs="?", default=False, help='If true, load pre-trained models and resume trainig from them using the new arguments.')
args = parser.parse_args()

t0 = time.time()

ttt.train(args.training_games, args.learnig_rate, args.random_move_prob, args.resume)
print("Number of training games: ", args.training_games)
print("Learning rate: ", args.learnig_rate)
print("Random move probability: ", args.random_move_prob)

t1 = time.time()
print("Training took", t1 - t0, "seconds.")
