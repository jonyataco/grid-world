# This program illustrates iterative policy evaluation

import numpy as np

# defining the parameters to be used
# we are using the grid that can be found on page 76 of the book.
# this is the discounting rate. since gridworld is episodic in nature,
# we are going to set it to 1
gamma = 1
GRID_SIZE = 4
terminationStates = [[0,0], [GRID_SIZE - 1, GRID_SIZE - 1]]
# the actions that are to be used, which correspond to:
# up, down, left, right.
actions = [[0,1], [0, -1], [-1,0], [1,0]]
iterations = 1000

"""
Returns the next state given an action.
The return value will be in the form of an array
that signifies the next state or position
"""
def performAction(initialState, action):
    nextState = np.array(initialState) + np.array(action)
    # This checks if the next state goes out of bounds.
    # This means that if the X or Y coordinate of the next state
    # will result in going out of bounds, then we stay in the same place
    if -1 in nextState or GRID_SIZE in nextState:
        nextState = initialState
    return nextState

"""
Function that returns a tuple of (nextState, reward)
If the initialState that is passed in is a terminal state,
then the agent does not move and the reward is zero.
"""
def determineStateReward(initialState, action):
    if initialState in terminationStates:
        return initialState, 0

    # Reward is going to be -1 on all transitions
    reward = -1
    nextState = performAction(initialState, action)
    return nextState, reward

# Initializing the values of each state in the grid to 0.
# Initializing the states as well.
value_grid = np.zeros((GRID_SIZE, GRID_SIZE))
states = [[i, j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]

# Going through 1000 iterations,
# and for each state we have to find V(s)
# which means finding the weight reward.
# To do this we get the reward from each action at each state
# Outerloop goes through each state in the grid and inner loop
# goes through each action for a given state
for iteration in range(iterations):
    for state in states:
        weightedReward = 0
        for action in actions:
            nextState, reward = determineStateReward(state, action)
            weightedReward += (1/len(actions))*(reward + (gamma * value_grid[nextState[0], nextState[1]]))
        value_grid[state[0], state[1]] = weightedReward
    if iteration in [0,1,2,9,10,20,50,99,iterations-1]:
        print("Iteration {}".format(iteration+1))
        print(value_grid)
        print("")
