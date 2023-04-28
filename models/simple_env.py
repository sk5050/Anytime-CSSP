import math
import time


class SimpleEnvModel(object):
    def __init__(
        self, start, goal, state_transition_probabilities, state_transition_costs
    ):
        self.init_state = start
        self.goal = goal
        self.state_transition_probabilities = state_transition_probabilities
        self.state_transition_costs = state_transition_costs

    def actions(self, state):
        if state == "start":
            return ["swp1", "swp2"]
        elif state == "wp1":
            return ["wp12", "wp1g"]
        elif state == "wp2":
            return ["wp21", "wp2g"]
        elif state == "goal" or state == "failure":
            return ["none"]

    def is_terminal(self, state):
        return state == self.goal

    def update_state_transition_costs(self, state_transition_costs):
        self.state_transition_costs = state_transition_costs

    def update_state_transition_probabilities(self, state_transition_probabilities):
        self.state_transition_probabilities = state_transition_probabilities

    def state_transitions(self, state, action):
        state_trans = []
        for next_state, prob in self.state_transition_probabilities[state][action]:
            state_trans.append([next_state, prob])
        return state_trans

    def cost(self, state, action):
        cost1 = self.state_transition_costs[state][action]
        cost2 = 0
        if state == "failure":
            cost2 = 1.0
        return cost1, cost2

    def heuristic(self, state):
        return 0, 0
