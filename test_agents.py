#!/usr/bin/env python3
import tic_tac_toe as ttt
import argparse

parser = argparse.ArgumentParser(
    description="""
    Test agents in 7 different settings.

    TTTSmartAgent(model1_SxS) vs TTTSmartAgent(model2_SxS)
    TTTSmartAgent(model1_SxD) vs TTTSmartAgent(model2_DxS)
    TTTSmartAgent(model1_SxS) vs TTTAgent()
    TTTSmartAgent(model1_SxD) vs TTTAgent()
    TTTAgent()                vs TTTSmartAgent(model2_SxS)
    TTTAgent()                vs TTTSmartAgent(model2_DxS)
    TTTAgent()                vs TTTAgent()
    """, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('num_trials', type=int, help='Number of games to be played at each setting.')
args = parser.parse_args()

ttt.test_agents(args.num_trials)
