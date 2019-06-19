# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # "*** YOUR CODE HERE ***"

        current_food = currentGameState.getFood()
        result = successorGameState.getScore()

        if action == Directions.STOP:
            return float("-inf")

        nearest_ghost = float("inf")
        for ghost in newGhostStates:
            dis_ghost = manhattanDistance(newPos, ghost.getPosition())
            if dis_ghost <= 1:
                return float("-inf")
            elif dis_ghost < nearest_ghost:
                nearest_ghost = dis_ghost

        nearest_food = float("inf")
        for x in range(1, current_food.width):
            for y in range(1, current_food.height):
                if current_food.data[x][y]:
                    dis_food = manhattanDistance(newPos, (x, y))
                    if dis_food < nearest_food:
                        nearest_food = dis_food

        return result - nearest_food + nearest_ghost + sum(newScaredTimes)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.max_level(gameState, 0)

    def max_level(self, gameState, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        max_next_level_scores = []
        pac_actions = gameState.getLegalActions(0)
        for pac_action in pac_actions:
            max_next_level_scores.append(self.min_level(gameState.generateSuccessor(0, pac_action), 1, depth))
        if depth == 0:
            max_index = max_next_level_scores.index(max(max_next_level_scores))
            return pac_actions[max_index]
        else:
            return max(max_next_level_scores)

    def min_level(self, gameState, ghost_index, depth):
        if gameState.isWin() or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState)

        min_next_level_scores = []
        ghost_actions = gameState.getLegalActions(ghost_index)
        for ghost_action in ghost_actions:
            if ghost_index < gameState.getNumAgents() - 1:
                min_next_level_scores.append(
                    self.min_level(gameState.generateSuccessor(ghost_index, ghost_action), ghost_index + 1, depth))
            else:
                min_next_level_scores.append(
                    self.max_level(gameState.generateSuccessor(ghost_index, ghost_action), depth + 1))
        return min(min_next_level_scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        initial_alpha = float("-inf")
        initial_beta = float("inf")
        return self.max_level(gameState, 0, initial_alpha, initial_beta)

    def max_level(self, gameState, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        max_next_level_scores = []
        pac_actions = gameState.getLegalActions(0)
        for pac_action in pac_actions:
            tmp = self.min_level(gameState.generateSuccessor(0, pac_action), 1, depth, alpha, beta)
            max_next_level_scores.append(tmp)
            max_so_far = max(max_next_level_scores)
            if max_so_far >= beta:
                return max_so_far
            alpha = max(alpha, max_so_far)
        if depth == 0:
            max_index = max_next_level_scores.index(max(max_next_level_scores))
            return pac_actions[max_index]
        else:
            return max(max_next_level_scores)

    def min_level(self, gameState, ghost_index, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState)

        min_next_level_scores = []
        ghost_actions = gameState.getLegalActions(ghost_index)
        for ghost_action in ghost_actions:
            if ghost_index < gameState.getNumAgents() - 1:
                tmp = self.min_level(gameState.generateSuccessor(ghost_index, ghost_action), ghost_index + 1, depth,
                                     alpha, beta)
                min_next_level_scores.append(tmp)
            else:
                tmp = self.max_level(gameState.generateSuccessor(ghost_index, ghost_action), depth + 1, alpha, beta)
                min_next_level_scores.append(tmp)
            min_so_far = min(min_next_level_scores)
            if min_so_far <= alpha:
                return min_so_far
            beta = min(beta, min_so_far)

        return min(min_next_level_scores)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        return self.max_level(gameState, 0)

    def max_level(self, gameState, depth):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        max_next_level_scores = []
        pac_actions = gameState.getLegalActions(0)
        for pac_action in pac_actions:
            max_next_level_scores.append(self.min_level(gameState.generateSuccessor(0, pac_action), 1, depth))
        if depth == 0:
            max_index = max_next_level_scores.index(max(max_next_level_scores))
            return pac_actions[max_index]
        else:
            return max(max_next_level_scores)

    def min_level(self, gameState, ghost_index, depth):
        if gameState.isWin() or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState)

        min_next_level_scores = []
        ghost_actions = gameState.getLegalActions(ghost_index)
        for ghost_action in ghost_actions:
            if ghost_index < gameState.getNumAgents() - 1:
                min_next_level_scores.append(
                    self.min_level(gameState.generateSuccessor(ghost_index, ghost_action), ghost_index + 1, depth))
            else:
                min_next_level_scores.append(
                    self.max_level(gameState.generateSuccessor(ghost_index, ghost_action), depth + 1))
        return sum(min_next_level_scores) / len(min_next_level_scores)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pac_position = currentGameState.getPacmanPosition()
    GhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    current_food = currentGameState.getFood()
    result = currentGameState.getScore()

    for x in range(1, current_food.width):
        for y in range(1, current_food.height):
            if current_food.data[x][y]:
                dis_food = manhattanDistance(pac_position, (x, y))
                if dis_food != 0:
                    result += (1 / dis_food)

    for ghost in GhostStates:
        dis_ghost = manhattanDistance(pac_position, ghost.getPosition())
        if dis_ghost <= 5 and dis_ghost != 0:
            result -= (1 / dis_ghost) * 5

    return result + (sum(newScaredTimes) * 5)


# Abbreviation
better = betterEvaluationFunction
