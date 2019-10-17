# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    Start: (5, 5)
    Is the start a goal? False
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    """
    "*** YOUR CODE HERE ***"
    visited = {}
    solution = []
    stack = util.Stack()
    route = {}

    start = problem.getStartState()
    stack.push((start, '', 0))
    visited[start] = ''
    if problem.isGoalState(start):
        return solution

    goal = False
    while not (stack.isEmpty() or goal):
        vertex = stack.pop()
        visited[vertex[0]] = vertex[1]
        if problem.isGoalState(vertex[0]):
            child = vertex[0]
            goal = True
            break
        for i in problem.getSuccessors(vertex[0]):
            if i[0] not in visited.keys():
                route[i[0]] = vertex[0]
                stack.push(i)

    while(child in route.keys()):
        parent = route[child]
        solution.insert(0, visited[child])
        child = parent

    return solution
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    solution = []
    queue = util.Queue()
    route = {}
    flag = False

    start = problem.getStartState()
    if problem.isGoalState(start):
      return solution
    queue.push((start, 'None', 0))
    visited[start] = 'None'

    while not (queue.isEmpty() or flag):
      vertex = queue.pop()
      visited[vertex[0]] = vertex[1]
      if problem.isGoalState(vertex[0]):
        child = vertex[0]
        flag = True
        break
      
      for i in problem.getSuccessors(vertex[0]):
        if i[0] not in visited.keys() and i[0] not in route.keys():
          route[i[0]] = vertex[0]
          queue.push(i)
    
    while (child in route.keys()):
      parent = route[child]
      solution.insert(0, visited[child])
      child = parent
    
    return solution
    if problem.isGoalState(start):
      return solution
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    solution = []
    queue = util.PriorityQueue()
    route = {}
    cost = {}

    start = problem.getStartState()
    queue.push((start, '', 0), 0)
    visited[start] = ''
    cost[start] = 0

    if problem.isGoalState(start):
        return solution

    flag = False
    while not (queue.isEmpty() or flag):
        vertex = queue.pop()
        visited[vertex[0]] = vertex[1]
        if problem.isGoalState(vertex[0]):
            child = vertex[0]
            flag = True
            break
        for i in problem.getSuccessors(vertex[0]):
            if i[0] not in visited.keys():
                priority = vertex[2] + i[2]
                if not(i[0] in cost.keys() and cost[i[0]] <= priority):
                    queue.push((i[0], i[1], vertex[2] + i[2]), priority)
                    cost[i[0]] = priority
                    route[i[0]] = vertex[0]

    while(child in route.keys()):
        parent = route[child]
        solution.insert(0, visited[child])
        child = parent

    return solution

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    solution = []
    queue = util.PriorityQueue()
    route = {}
    cost = {}

    start = problem.getStartState()
    queue.push((start, '', 0), 0)
    visited[start] = ''
    cost[start] = 0

    if problem.isGoalState(start):
        return solution

    flag = False
    while not (queue.isEmpty() or flag):
        vertex = queue.pop()
        visited[vertex[0]] = vertex[1]
        if problem.isGoalState(vertex[0]):
            child = vertex[0]
            flag = True
            break
        for i in problem.getSuccessors(vertex[0]):
            if i[0] not in visited.keys():
                priority = vertex[2] + i[2] + heuristic(i[0], problem)
                if not(i[0] in cost.keys() and cost[i[0]] <= priority):
                    queue.push((i[0], i[1], vertex[2] + i[2]), priority)
                    cost[i[0]] = priority
                    route[i[0]] = vertex[0]

    while(child in route.keys()):
        parents = route[child]
        solution.insert(0, visited[child])
        child = parents

    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
