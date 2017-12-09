#!/usr/bin/env python3
import tic_tac_toe as ttt
import argparse

parser = argparse.ArgumentParser(
    description="""
    Play against trained agents or watch them play against each other.""",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('agent1', type=str, help="Possible agents: SxS, SxD, human, random.")
parser.add_argument('agent2', type=str, help="Possible agents: SxS, DxS, human, random.")
args = parser.parse_args()

ttt.play_agent(args.agent1, args.agent2)
