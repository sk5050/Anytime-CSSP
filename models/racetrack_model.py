#!/usr/bin/env python

import math
import random
# import numpy as np
import time
import json
from collections import deque

class RaceTrackModel(object):

    def __init__(self, map_file, init_state, traj_check_dict_file=None, heuristic_file=None, slip_prob=0.1):


        self.init_state = init_state
        self.slip_prob = slip_prob
        
        self.action_list = []
        for x_val in [-1,0,1]:
            for y_val in [-1,0,1]:
                self.action_list.append((x_val, y_val))
                
        self.goal = (-1,-1)

        self.read_map(map_file)

        if traj_check_dict_file:
            with open(traj_check_dict_file) as f:
                self.traj_check_dict = json.load(f)

        else:
            self.traj_check_dict = None


        if heuristic_file:
            with open(heuristic_file) as f:
                self.heuristic_dict = json.load(f)

        else:
            self.heuristic_dict = None


        # self.traj_check_dict = dict()


    def read_map(self, map_file):

        self.ontrack_pos_set = []
        self.initial_pos_set = []
        self.finishline_pos_set = []
        self.constrained_pos_set = []

        map_text = open(map_file, 'r')
        lines = map_text.readlines()

        y = 0
        for line in reversed(lines):
            x = 0
            for char in line:
                if char==" ":
                    self.ontrack_pos_set.append((x,y))
                elif char=="s":
                    self.initial_pos_set.append((x,y))
                    self.ontrack_pos_set.append((x,y))  ## starting line is also considered as ontrack.
                elif char=="f":
                    self.finishline_pos_set.append((x,y))

                elif char=="x":
                    self.constrained_pos_set.append((x,y))

                x += 1

            y += 1

            

        
    def actions(self, state):
        return self.action_list

    
    def is_terminal(self, state):
        return state[0:2] == self.goal

    
    def state_transitions(self, state, action):

        new_state = (state[0]+state[2]+action[0], state[1]+state[3]+action[1], \
                     state[2]+action[0], state[3]+action[1])

        slip_state = (state[0]+state[2], state[1]+state[3],\
                      state[2], state[3])


        if self.traj_check_dict:
            if str((state[0],state[1],new_state[0],new_state[1])) in self.traj_check_dict:
                new_state_result = self.traj_check_dict[str((state[0],state[1],new_state[0],new_state[1]))]
                slip_state_result = self.traj_check_dict[str((state[0],state[1],slip_state[0],slip_state[1]))]
            else:
                new_state_result = self.bresenham_check_crash(state[0],state[1],new_state[0],new_state[1])
                slip_state_result = self.bresenham_check_crash(state[0],state[1],slip_state[0],slip_state[1])

        else:
            new_state_result = self.bresenham_check_crash(state[0],state[1],new_state[0],new_state[1])
            slip_state_result = self.bresenham_check_crash(state[0],state[1],slip_state[0],slip_state[1])

        # self.traj_check_dict[str((state[0],state[1],new_state[0],new_state[1]))] = new_state_result
        # self.traj_check_dict[str((state[0],state[1],slip_state[0],slip_state[1]))] = slip_state_result

        if new_state_result == "crash":
            rand_init_pos = self.init_state[0:2]
            # rand_init_pos = random.choice(self.initial_pos_set)
            new_state = rand_init_pos + (0,0)
            
        elif new_state_result == "finish":
            new_state = self.goal + (0,0)

        else:
            if new_state_result != "ontrack":
                raise ValueError("trajectory result is invalid!")


        if slip_state_result == "crash":
            rand_init_pos = self.init_state[0:2]
            # rand_init_pos = random.choice(self.initial_pos_set)
            slip_state = rand_init_pos + (0,0)
            
        elif slip_state_result == "finish":
            slip_state = self.goal + (0,0)

        else:
            if slip_state_result != "ontrack":
                raise ValueError("trajectory result is invalid!")

        if new_state==slip_state:
            new_states = [[new_state, 1.0]]
        else:
            new_states = [[new_state, 1-self.slip_prob], [slip_state, self.slip_prob]]

        return new_states




    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function. 
        cost1 = 1.0
        # if state[0:2] in self.initial_pos_set:
        # if state[1]<=1: ## simple
        # if state[0]>=26: ## easy
        # if state[1]<=2 or state[0]>=35: ## racetrack1
        # if state[1]>=30 or state[0]>=29: ## hard
        if (state[0], state[1]) in self.constrained_pos_set:
            cost2 = 10.0
        else:
            cost2 = 0.0
        return cost1, cost2

    
    def heuristic(self, state,depth=None):

        # heuristic1 = 0

        
        if self.heuristic_dict == None:
            heuristic1 = 0
        else:
            # if str(state) in self.heuristic_dict:
            heuristic1 = self.heuristic_dict[str(state)] - 1
            # else:
            # heuristic1 = 0
        
        heuristic2 = 0
        return heuristic1, heuristic2




    def check_traj(self, pos):

        if pos in self.ontrack_pos_set:
            return False
        elif pos in self.finishline_pos_set:
            return "finish"
        else:
            return "crash"


    def bresenham_check_crash(self, x1, y1, x2, y2):


        mark = []
        epsilon = 1e-1

        ### parallel cases

        dx = x2 - x1
        if dx==0:
            if y1<y2:
                for j in range(y1,y2+1):
                    mark.append((x1,j))

                    traj_result = self.check_traj(mark[-1])
                    if traj_result:
                        return traj_result
                    
            elif y1>y2:
                for j in range(y2,y1+1):
                    mark.append((x1,j))

                    traj_result = self.check_traj(mark[-1])
                    if traj_result:
                        return traj_result

            return "ontrack"


        dy = y2 - y1
        if dy==0:
            if x1<x2:
                for i in range(x1,x2+1):
                    mark.append((i,y1))

                    traj_result = self.check_traj(mark[-1])
                    if traj_result:
                        return traj_result
                    
            elif x1>x2:
                for i in range(x2,x1+1):
                    mark.append((i,y1))

                    traj_result = self.check_traj(mark[-1])
                    if traj_result:
                        return traj_result

            return "ontrack"
                    



        ### diagonal cases
        
        if abs(dx)==abs(dy):
            if x1<x2:
                if y1<y2:
                    for i in range(dx+1):
                        mark.append((x1+i, y1+i))

                        traj_result = self.check_traj(mark[-1])
                        if traj_result:
                            return traj_result                        

                elif y1>y2:
                    for i in range(dx+1):
                        mark.append((x1+i, y1-i))

                        traj_result = self.check_traj(mark[-1])
                        if traj_result:
                            return traj_result                        
                        
            elif x1>x2:
                if y1<y2:
                    for i in range(-dx+1):
                        mark.append((x1-i, y1+i))

                        traj_result = self.check_traj(mark[-1])
                        if traj_result:
                            return traj_result                        
                        
                        
                elif y1>y2:
                    for i in range(-dx+1):
                        mark.append((x1-i, y1-i))

                        traj_result = self.check_traj(mark[-1])
                        if traj_result:
                            return traj_result                        
                        

            return "ontrack"



        ### eight octants separately
        

        x1 = x1 + 0.5
        y1 = y1 + 0.5
        x2 = x2 + 0.5
        y2 = y2 + 0.5


        if abs(dy)>abs(dx):
            # print("|dy| > |dx|")
            m = abs(float(dx) / float(dy))
            i = int(x1)
            j1 = int(y1)
            j2 = int(y2)

            if dx > 0:
                # print("dx > 0")
                i_inc = 1
                if dy > 0:
                    e = -(1-(x1-i)-(1-(y1-j1))*m)                    

                    for j in range(j1,j2+1):

                        if abs(e)<epsilon:
                            i += i_inc
                            e -= 1.0
                            mark.append((i-1,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        
                            
                         

                        elif e>0:
                            i += i_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i-1,j))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        
                            

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        


                        e += m



                elif dy < 0:
                    e = -(1-(x1-i)-(y1-j1)*m)

                    for j in reversed(range(j2,j1+1)):

                        if abs(e)<epsilon:
                            i += i_inc
                            e -= 1.0
                            mark.append((i-1,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            i += i_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i-1,j))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result
                            
                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m                    


            else:
                # print("dx < 0")
                i_inc = -1
                if dy > 0:
                    e = -((x1-i)-(1-(y1-j1))*m)                    

                    for j in range(j1,j2+1):

                        if abs(e)<epsilon:
                            i += i_inc
                            e -= 1.0
                            mark.append((i+1,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            i += i_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i+1,j))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m                     



                elif dy < 0:
                    e = -((x1-i)-(y1-j1)*m)
                    for j in reversed(range(j2,j1+1)):

                        if abs(e)<epsilon:
                            i += i_inc
                            e -= 1.0
                            mark.append((i+1,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            i += i_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i+1,j))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m                        



        elif abs(dy)<abs(dx):
            # print("|dy| < |dx|")
            m = abs(float(dy) / float(dx))
            i1 = int(x1)
            i2 = int(x2)
            j = int(y1)

            if dy > 0:
                j_inc = 1
                if dx > 0:
                    e = -(1-(y1-j)-(1-(x1-i1))*m)

                    for i in range(i1,i2+1):

                        if abs(e)<epsilon:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j-1))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i,j-1))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m                    



                elif dx < 0:
                    e = -(1-(y1-j)-(x1-i1)*m)

                    for i in reversed(range(i2, i1+1)):

                        if abs(e)<epsilon:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j-1))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i,j-1))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m                      




            else:
                j_inc = -1
                if dx > 0:
                    e = -((y1-j)-(1-(x1-i1))*m)

                    for i in range(i1,i2+1):

                        if abs(e)<epsilon:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j+1))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i,j+1))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m


                elif dx < 0:
                    e = -((y1-j)-(x1-i1)*m)

                    for i in reversed(range(i2,i1+1)):

                        if abs(e)<epsilon:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j+1))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        elif e>0:
                            j += j_inc
                            e -= 1.0
                            mark.append((i,j))
                            mark.append((i,j+1))

                            traj_result = self.check_traj(mark[-2])
                            if traj_result:
                                return traj_result

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        else:
                            mark.append((i,j))

                            traj_result = self.check_traj(mark[-1])
                            if traj_result:
                                return traj_result                        

                        e += m                   

        return "ontrack"


    def heuristic_computation(self, graph):

        for state, node in graph.nodes.items():
            if state[0:2]==(-1,-1):
                goal_node = node
                break

            
        heuristic = dict()

        queue = deque([goal_node])
        visited = set()

        level = 0
        while queue:

            level += 1
            level_size = len(queue)

            while level_size > 0:

                node = queue.popleft()
                level_size -= 1
                
                if node in visited:
                    continue
                else:
                    for n in node.parents_set:
                        queue.append(n)
                    heuristic[str(node.state)] = level

                visited.add(node)

        return heuristic
