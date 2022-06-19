#!/usr/bin/env python

import math
# import numpy as np
import time

class LAOModel(object):

    def __init__(self):

        self.init_state=1
        self.goal = 4

        self.action_list = ["N","E","S"]

        
    def actions(self, state):
        return self.action_list

    
    def is_terminal(self, state):
        return state == self.goal

    
    def state_transitions(self, state, action):
        if action=="N":
            if state==1:
                new_states = [[1,0.5],[2,0.5]]
            elif state==2:
                new_states = [[2,1.0]]
            elif state==3:
                new_states = [[3,1.0]]
            elif state==4:
                new_states = [[4,1.0]]
            elif state==5:
                new_states = [[5,0.5],[4,0.5]]
            elif state==6:
                new_states = [[6,0.5],[5,0.5]]
            elif state==7:
                new_states = [[7,1.0]]
            elif state==8:
                new_states = [[8,0.5],[1,0.5]]

        elif action=="E":
            if state==1:
                new_states = [[1,1.0]]
            elif state==2:
                new_states = [[2,0.5],[3,0.5]]
            elif state==3:
                new_states = [[3,0.5],[4,0.5]]
            elif state==4:
                new_states = [[4,1.0]]
            elif state==5:
                new_states = [[5,1.0]]
            elif state==6:
                new_states = [[6,1.0]]
            elif state==7:
                new_states = [[7,0.5],[6,0.5]]
            elif state==8:
                new_states = [[8,0.5],[7,0.5]]

        elif action=="S":
            if state==1:
                new_states = [[1,0.5],[8,0.5]]
            elif state==2:
                new_states = [[2,0.5],[1,0.5]]
            elif state==3:
                new_states = [[3,1.0]]
            elif state==4:
                new_states = [[4,0.5],[5,0.5]]
            elif state==5:
                new_states = [[5,0.5],[6,0.5]]
            elif state==6:
                new_states = [[6,1.0]]
            elif state==7:
                new_states = [[7,1.0]]
            elif state==8:
                new_states = [[8,1.0]]

                
        return new_states


    def cost(self,state,action):
        return 1.0

    
    def heuristic(self, state,depth=None):
        if state==1:
            h = 3
        elif state==2:
            h = 2
        elif state==3:
            h = 1
        elif state==4:
            h = 0
        elif state==5:
            h = 1
        elif state==6:
            h = 2
        elif state==7:
            h = 3
        elif state==8:
            h = 4
            
        return h

