# Tic-Tac-Toe
Train agents to play Tic-Tac-Toe using Policy Gradient


Siraj Raval's Coding challenge 09/12/2017

Requirements : 
Python 3

How to use:

Run

$ ./train_agents.py training_games learning_rate
to train the agents, where 'training_games' is the number of training games to be played and 'learning_rate' is the gradient descent learning rate that updates the policy network.

After training, you can run

$ ./test_agents.py num_trials
to test the agents' performance against each other, where 'num_trials' is the number of games to be played by the agents.

You can also run

$ ./play_agent.py agent1 agent2
to play against a trained agent or watch them play against each other, where agent1 and agent2 are the types of the agent.

For more information, run

$ ./script_name.py -h
