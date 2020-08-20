import numpy as np
import random

# Defining the parameters to be used
# we are using the grid that can be found on page 76 of the book.
# this is the discounting rate. since gridworld is episodic in nature,
# we are going to set it to 1
gamma = .1
alpha = .1
GRID_SIZE = 4
rewardSize = -1
terminationStates = [[0,0], [GRID_SIZE - 1, GRID_SIZE - 1]]
# the actions that are to be used, which correspond to:
# up, down, left, right.
actions = [[0,1], [0, -1], [-1,0], [1,0]]
iterations = 2000
value_grid = np.zeros((GRID_SIZE,GRID_SIZE))
states = [[i,j] for i in range (GRID_SIZE) for j in range(GRID_SIZE)]

def takeAction(initialState, action):
    if list(initialState) in terminationStates:
        return None, 0
    nextState = np.array(initialState) + np.array(action)
    # If the action causes the agent to go out of bounds, then
    # have it stay in place.
    if -1 in list (nextState) or GRID_SIZE in list(nextState):
        nextState = initialState
    return list(nextState), rewardSize

for iteration in range(iterations):
    # The initial state will be a random selection of any of the states exluding
    # the terminal states. The action to be taken will be based on the policy,
    # which in this case is a random choice between up,down,left,right with
    # uniform probability
    initialState = random.choice(states[1:-1])
    while True:
        action = random.choice(actions)
        nextState, reward = takeAction(initialState, action)
        # Checks to see if we have reached a terminal state
        if reward == 0:
            break

        # Update the value of the initial state
        value_grid[initialState[0], initialState[1]] += alpha*(reward + gamma*value_grid[nextState[0], nextState[1]] - value_grid[initialState[0], initialState[1]])
        initialState = nextState

    if iteration in [0,1,2,9,10,20,50,99,iterations-1]:
        print("Iteration {}".format(iteration+1))
        print(value_grid)
        print("")
