#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph
from value_iteration import VI

import numpy as np

import gurobipy as gp
from gurobipy import GRB


class IDUAL(object):

    def __init__(self, model, bound):

        self.model = model
        self.bound = bound

        self.graph = Graph(name='G')
        self.graph.add_root(model.init_state, *self.model.heuristic(model.init_state))

        self.init_state = self.model.init_state

        self.lp_time = np.inf
        self.lp_nodes = np.inf



    def solve(self):

        S_hat = set([self.graph.root])
        F = set([self.graph.root])
        F_R = set([self.graph.root])
        F = set()
        G_hat = set()
        G = set() # this is a set for explored goals, i.e., S_hat.intersection(a set of all goals of the problem)

        while len(F_R)>0:

            N, G = self.F_R_expansion(F_R, G)   ## F_R_expansion not only expands fringe states, but also add explored goals to G. 
            # S_hat = S_hat.union(N)         ## we don't need S_hat explicitly, because S_hat=self.graph.nodes
            F_term_1 = F.difference(F_R)
            F_term_2 = N.difference(G)
            F = F_term_1.union(F_term_2)
            G_hat = F.union(G)
            F_R = self.solve_opt(G_hat, F)




    def solve_LP(self):

        S_hat = set([self.graph.root])
        F = set([self.graph.root])
        F_R = set([self.graph.root])
        F = set()
        G_hat = set()
        G = set() # this is a set for explored goals, i.e., S_hat.intersection(a set of all goals of the problem)

        while len(F_R)>0:

            N, G = self.F_R_expansion(F_R, G)   ## F_R_expansion not only expands fringe states, but also add explored goals to G. 
            # S_hat = S_hat.union(N)         ## we don't need S_hat explicitly, because S_hat=self.graph.nodes
            F_term_1 = F.difference(F_R)
            F_term_2 = N.difference(G)
            F = F_term_1.union(F_term_2)
            G_hat = F.union(G)
            F_R = self.solve_opt_LP(G_hat, F)            

            


    def solve_LP_and_MILP(self, t, time_limit=np.inf):

        S_hat = set([self.graph.root])
        F = set([self.graph.root])
        F_R = set([self.graph.root])
        F = set()
        G_hat = set()
        G = set() # this is a set for explored goals, i.e., S_hat.intersection(a set of all goals of the problem)

        t = time.time()
        while len(F_R)>0:

            N, G = self.F_R_expansion(F_R, G)   ## F_R_expansion not only expands fringe states, but also add explored goals to G. 
            # S_hat = S_hat.union(N)         ## we don't need S_hat explicitly, because S_hat=self.graph.nodes
            F_term_1 = F.difference(F_R)
            F_term_2 = N.difference(G)
            F = F_term_1.union(F_term_2)
            G_hat = F.union(G)
            F_R = self.solve_opt_LP(G_hat, F)

            print("-"*50)
            print(len(self.graph.nodes))
            print("elapsed time: "+str(time.time()-t))
            print("-"*50)

            if time.time() - t > time_limit:
                return np.inf, 1800, np.inf

        lp_time = time.time() - t
        self.lp_time = lp_time
        self.lp_nodes = len(self.graph.nodes)
            

        print("-"*50)
        print("-"*50)
        print("-"*50)
        print("LP finished")
        print("elapsed time: "+str(time.time()-t))
        print("number of states explored: "+str(len(self.graph.nodes)))
        print("-"*50)
        print("-"*50)
        print("-"*50)
        
        F_R, objVal = self.solve_opt(G_hat, F)

        while len(F_R)>0:

            N, G = self.F_R_expansion(F_R, G)   ## F_R_expansion not only expands fringe states, but also add explored goals to G. 
            # S_hat = S_hat.union(N)         ## we don't need S_hat explicitly, because S_hat=self.graph.nodes
            F_term_1 = F.difference(F_R)
            F_term_2 = N.difference(G)
            F = F_term_1.union(F_term_2)
            G_hat = F.union(G)
            F_R, objVal = self.solve_opt(G_hat, F)

            print("-"*50)
            print(len(self.graph.nodes))
            print("elapsed time: "+str(time.time()-t))
            print("-"*50)

            # if time.time() - t > time_limit:
            return np.inf, self.lp_time, self.lp_nodes


        print("-"*50)
        print("-"*50)
        print("-"*50)
        print("MILP finished")
        print("elapsed time: "+str(time.time()-t))
        print("number of states explored: "+str(len(self.graph.nodes)))
        print("-"*50)
        print("-"*50)
        print("-"*50)

        return objVal, self.lp_time, self.lp_nodes




    def F_R_expansion(self, F_R, G):
        N = set()
        for node in F_R:
            new_N, new_G = self.expand(node)
            N = N.union(new_N)
            G = G.union(new_G)

        return N, G


    def expand(self, expanded_node):
        new_G = set()
        new_N = set()
        
        state = expanded_node.state
        actions = self.model.actions(state)

        for action in actions:
            children = self.model.state_transitions(state,action)

            children_list = []
            for child_state, child_prob in children:
                if child_state in self.graph.nodes:
                    child = self.graph.nodes[child_state]
                else:
                    self.graph.add_node(child_state, self.model.heuristic(child_state))
                    child = self.graph.nodes[child_state]

                    new_N.add(child)

                    if self.model.is_terminal(child.state):
                        child.set_terminal()
                
                child.parents_set.add(expanded_node)
                children_list.append([child,child_prob])

                if child.terminal==True:
                    new_G.add(child)

            expanded_node.children[action] = children_list

        return new_N, new_G
        



    def solve_opt(self, G_hat, F):

        state_list = []
        goal_list  = []

        for state,node in self.graph.nodes.items():

            if node in G_hat:
                goal_list.append(state)
            else:
                state_list.append(state)


        m = gp.Model("CSSP")

        X = 100000


        ## Add variables
        state_var_dict = dict()
        goal_var_dict = dict()

        delta_state_var_dict = dict()
        delta_goal_var_dict = dict()

        for state in state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(lb=0, name=str((state,action)))

            state_var_dict[state] = dict_temp

        for goal in goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(lb=0, name=str((goal,action)))

            goal_var_dict[goal] = dict_temp



        for state in state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(vtype=GRB.BINARY)

            delta_state_var_dict[state] = dict_temp

        for goal in goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(vtype=GRB.BINARY)

            delta_goal_var_dict[goal] = dict_temp



        in_var_dict = dict()
        out_var_dict = dict()

        for state in state_list:
            in_var_dict[state] = m.addVar(lb=0, name=str(state))
            out_var_dict[state] = m.addVar(lb=0, name=str(state))

        for goal in goal_list:
            in_var_dict[goal] = m.addVar(lb=0, name=str(goal))

            
        ## Add constraints

        for state in state_list:
            in_var = in_var_dict[state]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,state,action_) \
                                for parent_node in self.graph.nodes[state].parents_set for action_ in self.model.actions(parent_node.state))
                        )


            out_var = out_var_dict[state]
            var_dict = state_var_dict[state]
            m.addConstr(out_var == \
                        gp.quicksum(var for action,var in var_dict.items()))
                        
        


        for goal in goal_list:
            in_var = in_var_dict[goal]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,goal,action_) \
                                for parent_node in self.graph.nodes[goal].parents_set for action_ in self.model.actions(parent_node.state))
                        )

            
                

        for state in state_list:
            if state==self.init_state:
                m.addConstr(
                    out_var_dict[state] == 1 + in_var_dict[state])

                
            else:
                m.addConstr(
                    out_var_dict[state] == in_var_dict[state])
            

        m.addConstr(
            0.9 <= gp.quicksum(in_var_dict[goal] for goal in goal_list))

        m.addConstr(
            1.1 >= gp.quicksum(in_var_dict[goal] for goal in goal_list))

        for i in range(len(self.bound)):

            m.addConstr(gp.quicksum(state_var_dict[state][action]*self.secondary_cost(state,action,i) \
                                    for state in state_list for action in self.model.actions(state)) + \
                        gp.quicksum(in_var_dict[goal]*self.secondary_heuristic(goal,i) for goal in goal_list)
                        <= self.bound[i])


        for state in state_list:
            m.addConstr(
                gp.quicksum(delta_state_var_dict[state][action] for action in self.model.actions(state))
                <= 1)

        for goal in goal_list:
            m.addConstr(
                gp.quicksum(delta_goal_var_dict[goal][action] for action in self.model.actions(goal))
                <= 1)


        for state in state_list:
            for action in self.model.actions(state):
                m.addConstr(
                    state_var_dict[state][action] / X <= delta_state_var_dict[state][action])

        for goal in goal_list:
            for action in self.model.actions(goal):
                m.addConstr(
                    goal_var_dict[goal][action] / X <= delta_goal_var_dict[goal][action])


        
        ## Add objective

        m.setObjective(gp.quicksum(state_var_dict[state][action]*self.primary_cost(state,action) \
                                   for state in state_list for action in self.model.actions(state)) + \
                       gp.quicksum(in_var_dict[goal]*self.primary_heuristic(goal) for goal in goal_list))     


        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        print('Obj: %g' % m.objVal)

        F_R = set()

        for node in F:
            state = node.state

            if in_var_dict[state].x > 0:
                F_R.add(node)

        

        return F_R, m.objVal






    def solve_opt_LP(self, G_hat, F):

        state_list = []
        goal_list  = []

        for state,node in self.graph.nodes.items():

            if node in G_hat:
                goal_list.append(state)
            else:
                state_list.append(state)


        m = gp.Model("CSSP")

        X = 100000


        ## Add variables
        state_var_dict = dict()
        goal_var_dict = dict()

        for state in state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(lb=0, name=str((state,action)))

            state_var_dict[state] = dict_temp

        for goal in goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(lb=0, name=str((goal,action)))

            goal_var_dict[goal] = dict_temp


        in_var_dict = dict()
        out_var_dict = dict()

        for state in state_list:
            in_var_dict[state] = m.addVar(lb=0, name=str(state))
            out_var_dict[state] = m.addVar(lb=0, name=str(state))

        for goal in goal_list:
            in_var_dict[goal] = m.addVar(lb=0, name=str(goal))

            
        ## Add constraints

        for state in state_list:
            in_var = in_var_dict[state]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,state,action_) \
                                for parent_node in self.graph.nodes[state].parents_set for action_ in self.model.actions(parent_node.state))
                        )


            out_var = out_var_dict[state]
            var_dict = state_var_dict[state]
            m.addConstr(out_var == \
                        gp.quicksum(var for action,var in var_dict.items()))
                        
        


        for goal in goal_list:
            in_var = in_var_dict[goal]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,goal,action_) \
                                for parent_node in self.graph.nodes[goal].parents_set for action_ in self.model.actions(parent_node.state))
                        )

            
                

        for state in state_list:
            if state==self.init_state:
                m.addConstr(
                    out_var_dict[state] == 1 + in_var_dict[state])

                
            else:
                m.addConstr(
                    out_var_dict[state] == in_var_dict[state])
            

        m.addConstr(
            0.9 <= gp.quicksum(in_var_dict[goal] for goal in goal_list))

        m.addConstr(
            1.1 >= gp.quicksum(in_var_dict[goal] for goal in goal_list))


        for i in range(len(self.bound)):

            m.addConstr(gp.quicksum(state_var_dict[state][action]*self.secondary_cost(state,action,i) \
                                    for state in state_list for action in self.model.actions(state)) + \
                        gp.quicksum(in_var_dict[goal]*self.secondary_heuristic(goal,i) for goal in goal_list)
                        <= self.bound[i])

        ## Add objective

        m.setObjective(gp.quicksum(state_var_dict[state][action]*self.primary_cost(state,action) \
                                   for state in state_list for action in self.model.actions(state)) + \
                       gp.quicksum(in_var_dict[goal]*self.primary_heuristic(goal) for goal in goal_list))     


        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        print('Obj: %g' % m.objVal)

        F_R = set()

        for node in F:
            state = node.state

            if in_var_dict[state].x > 0:
                F_R.add(node)

        return F_R


    def prob(self, from_state, to_state, action):

        new_states = self.model.state_transitions(from_state, action)

        for new_state in new_states:
            if new_state[0]==to_state:
                return new_state[1]

        return 0.0

    

    def primary_cost(self, state, action):
        if len(self.bound)==1:
            cost_1, cost_2 = self.model.cost(state,action)
        elif len(self.bound)==2:
            cost_1, cost_2, cost_3 = self.model.cost(state,action)
        elif len(self.bound)==3:
            cost_1, cost_2, cost_3, cost_4 = self.model.cost(state,action)
        elif len(self.bound)==4:
            cost_1, cost_2, cost_3, cost_4, cost_5 = self.model.cost(state,action)
        elif len(self.bound)==5:
            cost_1, cost_2, cost_3, cost_4, cost_5, cost_6 = self.model.cost(state,action)
        elif len(self.bound)==6:
            cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7 = self.model.cost(state,action)
            
        return cost_1

    def secondary_cost(self, state, action, num):
        if len(self.bound)==1:
            cost_1, cost_2 = self.model.cost(state,action)
        elif len(self.bound)==2:
            cost_1, cost_2, cost_3 = self.model.cost(state,action)
        elif len(self.bound)==3:
            cost_1, cost_2, cost_3, cost_4 = self.model.cost(state,action)
        elif len(self.bound)==4:
            cost_1, cost_2, cost_3, cost_4, cost_5 = self.model.cost(state,action)
        elif len(self.bound)==5:
            cost_1, cost_2, cost_3, cost_4, cost_5, cost_6 = self.model.cost(state,action)
        elif len(self.bound)==6:
            cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7 = self.model.cost(state,action)

        if num==0:
            return cost_2
        elif num==1:
            return cost_3
        elif num==2:
            return cost_4
        elif num==3:
            return cost_5
        elif num==4:
            return cost_6
        elif num==5:
            return cost_7

    def primary_heuristic(self, state):
        if len(self.bound)==1:
            heuristic_1, heuristic_2 = self.model.heuristic(state)
        elif len(self.bound)==2:
            heuristic_1, heuristic_2, heuristic_3 = self.model.heuristic(state)
        elif len(self.bound)==3:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4 = self.model.heuristic(state)
        elif len(self.bound)==4:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5 = self.model.heuristic(state)
        elif len(self.bound)==5:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5, heuristic_6 = self.model.heuristic(state)
        elif len(self.bound)==6:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5, heuristic_6, heuristic_7 = self.model.heuristic(state)
            
        return heuristic_1

    def secondary_heuristic(self, state, num):
        if len(self.bound)==1:
            heuristic_1, heuristic_2 = self.model.heuristic(state)
        elif len(self.bound)==2:
            heuristic_1, heuristic_2, heuristic_3 = self.model.heuristic(state)
        elif len(self.bound)==3:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4 = self.model.heuristic(state)
        elif len(self.bound)==4:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5 = self.model.heuristic(state)
        elif len(self.bound)==5:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5, heuristic_6 = self.model.heuristic(state)
        elif len(self.bound)==6:
            heuristic_1, heuristic_2, heuristic_3, heuristic_4, heuristic_5, heuristic_6, heuristic_7 = self.model.heuristic(state)

        if num==0:
            return heuristic_2
        elif num==1:
            return heuristic_3
        elif num==2:
            return heuristic_4
        elif num==3:
            return heuristic_5
        elif num==4:
            return heuristic_6
        elif num==5:
            return heuristic_7
