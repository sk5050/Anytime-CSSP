class ObstacleEnvModel(object):
    def __init__(self, skills, start, goal, state_transition_probabilities, state_transition_costs, aux_action="none"):
        self.skills = skills
        self.init_state = start
        self.goal_state = goal
        self.auxilliary_action = aux_action
        self.state_transition_costs = state_transition_costs
        self.state_transition_probabilities = state_transition_probabilities

    def actions(self, state):
        if state != "goal" and state != "failure":
            return self.skills[state]
        else:
            return [self.auxilliary_action]

    def is_terminal(self, state):
        return state == self.goal_state

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
