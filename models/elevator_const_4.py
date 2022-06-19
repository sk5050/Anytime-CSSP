#!/usr/bin/env python

import math
import time

class ELEVATORModel_2(object):

    def __init__(self, n=20, w=1, h=1, prob=0.75, init_state=None, px_dest=None, hidden_dest=None, hidden_origin=None):

        self.init_state=init_state
        self.n = n
        self.w = w
        self.h = h
        self.prob = prob
        self.px_dest = px_dest
        self.hidden_dest = hidden_dest
        self.hidden_origin = hidden_origin
        # self.goal = goal

        self.action_list = ["OO","CO","UO","DO","OC","CC","UC","DC","OU","CU","UU","DU","OD","CD","UD","DD"]

        
    def actions(self, state):

        if state[2][1]==1 and state[3][1]==1:
            return ['CC']
        elif state[2][1]==1:
            return ['CO', 'CU', 'CD']
        elif state[3][1]==1:
            return ['OC', 'UC', 'DC']
        else:
            return ['OO','UO','DO', 'OU', 'UU', 'DU', 'OD', 'UD', 'DD']

    
    def is_terminal(self, state):

        try:
            if max(state[0]) == -2 and max(state[1]) == -2:
                return True
            else:
                return False

        except:
            return False

        
    def state_transitions(self, state, action):

        px_at = state[0]
        hidden_at = state[1]
        elev_1 = state[2]
        elev_2 = state[3]

        new_px_pos = []
        for px_ind in range(self.w):
            px_pos = px_at[px_ind]
            if px_pos==-2:
                new_px_pos.append(px_pos)
                continue
            elif px_pos==-10:
                if elev_1[0]==self.px_dest[px_ind] and action[0]=='O':
                    new_px_pos.append(-2)
                else:
                    new_px_pos.append(px_pos)
            elif px_pos==-20:
                if elev_2[0]==self.px_dest[px_ind] and action[1]=='O':
                    new_px_pos.append(-2)
                else:
                    new_px_pos.append(px_pos)
            else:
                if elev_1[0]==px_pos and action[0]=='O':
                    new_px_pos.append(-10)
                elif elev_2[0]==px_pos and action[1]=='O':
                    new_px_pos.append(-20)
                else:
                    new_px_pos.append(px_pos)


        new_hidden_pos = []
        unarrived_hidden_ind_list = []
        for hidden_ind in range(self.h):
            hidden_pos = hidden_at[hidden_ind]
            if hidden_pos==0:
                unarrived_hidden_ind_list.append(hidden_ind)
                new_hidden_pos.append(0)
                continue

            elif hidden_pos==-2:
                new_hidden_pos.append(hidden_pos)
                continue
            elif hidden_pos==-10:
                if elev_1[0]==self.hidden_dest[hidden_ind] and action[0]=='O':
                    new_hidden_pos.append(-2)
                else:
                    new_hidden_pos.append(hidden_pos)
            elif hidden_pos==-20:
                if elev_2[0]==self.hidden_dest[hidden_ind] and action[1]=='O':
                    new_hidden_pos.append(-2)
                else:
                    new_hidden_pos.append(hidden_pos)
            else:
                if elev_1[0]==hidden_pos and action[0]=='O':
                    new_hidden_pos.append(-10)
                elif elev_2[0]==hidden_pos and action[1]=='O':
                    new_hidden_pos.append(-20)
                else:
                    new_hidden_pos.append(hidden_pos)


        ## elev 1
        if action[0]=='U':
            new_elev_1_pos = min(self.n, elev_1[0] + 1)
        elif action[0]=='D':
            new_elev_1_pos = max(1, elev_1[0] - 1)
        else:
            new_elev_1_pos = elev_1[0]

        if action[0]=='O':
            new_elev_1_is_open = 1
        elif action[0]=='C':
            new_elev_1_is_open = 0
        else:
            new_elev_1_is_open = elev_1[1]

        new_elev_1 = (new_elev_1_pos, new_elev_1_is_open)


        ## elev 2
        if action[1]=='U':
            new_elev_2_pos = min(self.n, elev_2[0] + 1)
        elif action[1]=='D':
            new_elev_2_pos = max(1, elev_2[0] - 1)
        else:
            new_elev_2_pos = elev_2[0]

        if action[1]=='O':
            new_elev_2_is_open = 1
        elif action[1]=='C':
            new_elev_2_is_open = 0
        else:
            new_elev_2_is_open = elev_2[1]

        new_elev_2 = (new_elev_2_pos, new_elev_2_is_open)
            

        new_states = [[[new_px_pos, new_hidden_pos, new_elev_1, new_elev_2], 1.0]]

        for unarrived_hidden_ind in unarrived_hidden_ind_list:
            new_states_temp = []
            for new_state in new_states:
                for trans in ['NA', 'A']:
                    new_hidden_pos_temp = new_state[0][1].copy()
                    if trans=='NA':
                        new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_elev_1, new_elev_2], new_state[1]*0.25])
                    else:
                        new_hidden_pos_temp[unarrived_hidden_ind] = self.hidden_origin[unarrived_hidden_ind]
                        new_states_temp.append([[new_px_pos, new_hidden_pos_temp, new_elev_1, new_elev_2], new_state[1]*0.75])

            new_states = new_states_temp
                


        new_states_in_tuple = []
        for new_state in new_states:
            new_states_in_tuple.append([(tuple(new_state[0][0]), tuple(new_state[0][1]), new_state[0][2], new_state[0][3]), new_state[1]])

  

        # print("----------------")
        # print(state)
        # print(action)
        # print(new_states)
        # time.sleep(1)
            
        return new_states_in_tuple
    

    


    def cost(self,state,action):  # cost function should return vector of costs, even though there is a single cost function.
        
        cost1 = 1.0

        if state[0][0]>0:
            cost2 = 1.0
        else:
            cost2 = 0.0

        if state[0][0]<=-10:
            cost3 = 1.0
        else:
            cost3 = 0.0

        if state[0][1]>0:
            cost4 = 1.0
        else:
            cost4 = 0.0

        if state[0][1]<=-10:
            cost5 = 1.0
        else:
            cost5 = 0.0

        if state[1][0]>0:
            cost6 = 1.0
        else:
            cost6 = 0.0

        if state[1][0]<=-10:
            cost7 = 1.0
        else:
            cost7 = 0.0

            

        return cost1, cost2, cost4, cost6, cost7

    
    def heuristic(self, state,depth=None):
        
        px_pos = state[0]
        hidden_pos = state[1]
        elev_1 = state[2]
        elev_2 = state[3]

        px_1_w = 0
        px_1_t = 0
        px_2_w = 0
        px_2_t = 0
        h_1_w = 0
        h_1_t = 0


        if px_pos[0]>0:
            px_1_w = min(abs(elev_1[0] - px_pos[0]), abs(elev_2[0] - px_pos[0]))

        if px_pos[0]==-10:
            px_1_t = abs(self.px_dest[0]-elev_1[0])
        elif px_pos[0]==-20:
            px_1_t = abs(self.px_dest[0]-elev_2[0])
        elif px_pos[0]>0:
            px_1_t = abs(self.px_dest[0]-px_pos[0])
        elif px_pos[0]==-2:
            px_1_t = 0


        if px_pos[1]>0:
            px_2_w = min(abs(elev_1[0] - px_pos[1]), abs(elev_2[0] - px_pos[1]))

        if px_pos[1]==-10:
            px_2_t = abs(self.px_dest[1]-elev_1[0])
        elif px_pos[1]==-20:
            px_2_t = abs(self.px_dest[1]-elev_2[0])
        elif px_pos[1]>0:
            px_2_t = abs(self.px_dest[1]-px_pos[1])
        elif px_pos[1]==-2:
            px_2_t = 0


        if hidden_pos[0]>0:
            hidden_1_w = min(abs(elev_1[0] - hidden_pos[0]), abs(elev_2[0] - hidden_pos[0]))

        if hidden_pos[0]==-10:
            hidden_1_t = abs(self.hidden_dest[0]-elev_1[0])
        elif hidden_pos[0]==-20:
            hidden_1_t = abs(self.hidden_dest[0]-elev_2[0])
        elif hidden_pos[0]>0:
            hidden_1_t = abs(self.hidden_dest[0]-hidden_pos[0])
        elif hidden_pos[0]==-2:
            hidden_1_t = 0

            

        h_cost = max(px_1_w+px_1_t, px_2_w+px_2_t, h_1_w+h_1_t)
        h_cost = max(0, h_cost-1)
            
        
        return h_cost, px_1_w, px_2_w, h_1_w, h_1_t
