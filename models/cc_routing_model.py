#!/usr/bin/env python

import math
import time

class CCROUTINGModel(object):

    def __init__(self, size=(10,10),obs_num=1,obs_dir=['U'],obs_boundary=[(3,3)], init_state=((0,0),(5,5)), goal=(9,9), prob_right_transition=0.9):


        self.init_state=init_state
        self.prob_right_transition = prob_right_transition
        self.size = size
        self.state_list = []
        self.goal = goal
        self.obs_dir=obs_dir
        self.obs_boundary = obs_boundary

        self.obs_num = obs_num

        for i in range(size[0]):
            for j in range(size[1]):
                self.state_list.append((i,j))

        self.action_list = ["U","L","R","D"]

        
    def actions(self, state):
        ac_state = state[0]

        for i in range(self.obs_num):

            if abs(ac_state[0]-state[i+1][0]) <= self.obs_boundary[i][0] and abs(ac_state[1]-state[i+1][1]) <= self.obs_boundary[i][1]:
                return ['failed']
        
        action_copied = self.action_list.copy()

        if ac_state[0]==0:
            action_copied.remove('L')
        elif ac_state[0]==self.size[0]-1:
            action_copied.remove('R')
            
        if ac_state[1]==0:
            action_copied.remove('D')
        elif ac_state[1]==self.size[1]-1:
            action_copied.remove('U')

        return action_copied

    
    def is_terminal(self, state):
        return state[0] == self.goal

    
    def state_transitions(self, state, action):
        ac_state = state[0]
        if action=="U":
            new_ac_state = (ac_state[0], ac_state[1]+1)
            
        elif action=="D":
            new_ac_state = (ac_state[0], ac_state[1]-1)

        elif action=="L":
            new_ac_state = (ac_state[0]-1, ac_state[1])

        elif action=="R":
            new_ac_state = (ac_state[0]+1, ac_state[1])

        elif action=="failed":
            new_ac_state = self.goal


        if new_ac_state == self.goal:
            if self.obs_num==1:
                return [[(self.goal,(0,0)),1.0]]
            elif self.obs_num==2:
                return [[(self.goal,(0,0),(0,0)),1.0]]
            else:
                raise ValueError("something wrong.")

            

            
            
        obs_states_total_temp = []
        
        for i in range(self.obs_num):

            obs_state = state[i+1]

            if obs_state == (-10,-10):
                obs_states_temp_2 = [[(-10,-10), 1.0]]

            else:

                if self.obs_dir[i]=='U':
                    obs_states_temp = [[(obs_state[0],obs_state[1]+1), self.prob_right_transition],
                                       [(obs_state[0]+1,obs_state[1]), (1-self.prob_right_transition)/2],
                                       [(obs_state[0]-1,obs_state[1]), (1-self.prob_right_transition)/2]]

                elif self.obs_dir[i]=='D':
                    obs_states_temp = [[(obs_state[0],obs_state[1]-1), self.prob_right_transition],
                                       [(obs_state[0]+1,obs_state[1]), (1-self.prob_right_transition)/2],
                                       [(obs_state[0]-1,obs_state[1]), (1-self.prob_right_transition)/2]]

                elif self.obs_dir[i]=='L':
                    obs_states_temp = [[(obs_state[0]-1,obs_state[1]), self.prob_right_transition],
                                       [(obs_state[0],obs_state[1]+1), (1-self.prob_right_transition)/2],
                                       [(obs_state[0],obs_state[1]-1), (1-self.prob_right_transition)/2]]

                elif self.obs_dir[i]=='R':
                    obs_states_temp = [[(obs_state[0]+1,obs_state[1]), self.prob_right_transition],
                                       [(obs_state[0],obs_state[1]+1), (1-self.prob_right_transition)/2],
                                       [(obs_state[0],obs_state[1]-1), (1-self.prob_right_transition)/2]]


                obs_states_temp_2 = []
                for obs_state in obs_states_temp:
                    
                    if obs_state[0][0]<0 or obs_state[0][0]>self.size[0]-1 or \
                       obs_state[0][1]<0 or obs_state[0][1]>self.size[1]-1:
                        obs_states_temp_2.append([(-10,-10), obs_state[1]])

                    else:
                        obs_states_temp_2.append(obs_state)

            obs_states_total_temp.append(obs_states_temp_2)


        new_states = []


        if self.obs_num==1:

            for obs_state_1 in obs_states_total_temp[0]:
                obs_pos_1 = obs_state_1[0]
                obs_prob_1 = obs_state_1[1]

                new_state = [(new_ac_state, obs_pos_1), obs_prob_1]
                new_states.append(new_state)
            

        elif self.obs_num==2:

            for obs_state_1 in obs_states_total_temp[0]:

                obs_pos_1 = obs_state_1[0]
                obs_prob_1 = obs_state_1[1]
                for obs_state_2 in obs_states_total_temp[1]:
                    obs_pos_2 = obs_state_2[0]
                    obs_prob_2 = obs_state_2[1]

                    new_state = [(new_ac_state, obs_pos_1, obs_pos_2), obs_prob_1*obs_prob_2]
                    new_states.append(new_state)


        elif self.obs_num==3:

            for obs_state_1 in obs_states_total_temp[0]:

                obs_pos_1 = obs_state_1[0]
                obs_prob_1 = obs_state_1[1]
                for obs_state_2 in obs_states_total_temp[1]:
                    obs_pos_2 = obs_state_2[0]
                    obs_prob_2 = obs_state_2[1]

                    for obs_state_3 in obs_states_total_temp[2]:
                        obs_pos_3 = obs_state_3[0]
                        obs_prob_3 = obs_state_3[1]

                        new_state = [(new_ac_state, obs_pos_1, obs_pos_2, obs_pos_3), obs_prob_1*obs_prob_2*obs_prob_3]
                        new_states.append(new_state)


        return new_states





    def cost(self,state,action):
        cost1 = 1.0
        cost2 = 0.0

        if action=='failed':
            cost1 = 0.0
            cost2 = 1.0

                
        return cost1, cost2
    
    
    def heuristic(self, state,depth=None):
        heuristic1 = math.sqrt((state[0][0]-self.goal[0])**2 + (state[0][1]-self.goal[1])**2)

        heuristic2 = 0

        return heuristic1, heuristic2


