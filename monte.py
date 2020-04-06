import numpy as np
import random

# Defining the parameters to be used
# we are using the grid that can be found on page 76 of the book.
# this is the discounting rate. since gridworld is episodic in nature,
# we are going to set it to 1
gamma = 1
GRID_SIZE = 4
rewardSize = -1
terminationStates = [[0,0], [GRID_SIZE - 1, GRID_SIZE - 1]]
# the actions that are to be used, which correspond to:
# up, down, left, right.
actions = [[0,1], [0, -1], [-1,0], [1,0]]
iterations = 1000

# Initializing the states for the grid, values of the grid, and returns
# for each state.
value_grid = np.zeros((GRID_SIZE, GRID_SIZE))
states = [[i, j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
returns = {(i,j):list() for i in range(GRID_SIZE) for j in range(GRID_SIZE)}
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
    if -1 in list(nextState) or GRID_SIZE in list(nextState):
        nextState = initialState
    return nextState

"""
Function that returns an episode.
An episode is simply an array where each index contains a subarray
of information. Each index of the episode is a timestep for the episode.
The subarray will contain the following information
[initialState (comes in the form of coordinates or tuple such as [0,2]),
 action (the action that was taken),
 rewardSize (the reward that was received),
 nextState (the state that comes after the action that was taken)
]
"""
def generateEpisode():
    # Will choose a state at random excluding the terminal states, which are
    # [0,0] and [3,3].
    initialState = random.choice(states[1:-1])
    episode = []
    while True:
        # Break out of the loop once the episode has reached a terminal state
        if list(initialState) in terminationStates:
            return episode
        # Randomly choose an action based on a uniform distribution.
        # Since there are 4 actions, each has a chance of being picked .25
        action = random.choice(actions)
        # Next state will be assigned based on the action and inital state.
        nextState = performAction(initialState, action)
        episode.append([list(initialState), action, rewardSize, list(nextState)])
        initialState = nextState

# Go through the specified number of iterations
for iteration in range(iterations):
    episode = generateEpisode()
    G = 0
    # Doing episode[::-1] reverses the episode array.
    # Passing this to enumerate allows us to start loop through
    # each step of the episode till we reach 0.
    for i, step in enumerate(episode[::-1]):
        # We are indexing a certain time step in the episode,
        # and by indexing step[2] we are accesing the reward.
        G = (gamma * G) + step[2]
        # Since this is first-visit monte carlo prediction,
        # We check if the initial state occured earlier in the episode
        # step[0] refers to the initalState
        if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
            # Since the initial state did not appear earlier, append G
            # to the returns dictionary at the specified key, in this case
            # the specified state
            state = (step[0][0], step[0][1])
            returns[state].append(G)
            value_grid[state[0],state[1]] = np.average(returns[state])

    if iteration in [0,1,2,9,10,20,50,99,iterations-1]:
        print("Iteration {}".format(iteration+1))
        print(value_grid)
        print("")
