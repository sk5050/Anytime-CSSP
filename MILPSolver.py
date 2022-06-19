#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph
from value_iteration import VI

import gurobipy as gp
from gurobipy import GRB


class MILPSolver(object):

    def __init__(self, model, bound, algo=None):

        self.model = model
        self.bound = bound

        if algo==None:
            self.algo = VI(model,constrained=True)
            self.algo.expand_all()
            print(len(self.algo.graph.nodes))
        else:
            self.algo = algo

        self.init_state = self.model.init_state
        self.state_list = []
        self.goal_list = []

        for state,node in self.algo.graph.nodes.items():

            if node.terminal==True:
                self.goal_list.append(state)

            else:
                self.state_list.append(state)



    def encode_MILP(self):

        m = gp.Model("CSSP")

        X = 100000


        ## Add variables
        state_var_dict = dict()
        goal_var_dict = dict()

        delta_state_var_dict = dict()
        delta_goal_var_dict = dict()

        for state in self.state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                # dict_temp[action] = m.addVar(vtype=GRB.BINARY, name=str((state,action)))
                dict_temp[action] = m.addVar(lb=0, name=str((state,action)))

            state_var_dict[state] = dict_temp

        for goal in self.goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                # dict_temp[action] = m.addVar(vtype=GRB.BINARY, name=str((goal,action)))
                dict_temp[action] = m.addVar(lb=0, name=str((goal,action)))

            goal_var_dict[goal] = dict_temp



        for state in self.state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(vtype=GRB.BINARY)

            delta_state_var_dict[state] = dict_temp

        for goal in self.goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(vtype=GRB.BINARY)

            delta_goal_var_dict[goal] = dict_temp



            

            
        ## Add constraints

        for state in self.state_list:
            if state==self.init_state:
                var_dict = state_var_dict[state]
                m.addConstr(
                    gp.quicksum(var for action,var in var_dict.items()) == \
                    1 + gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,state,action_) \
                                for parent_node in self.algo.graph.nodes[state].parents_set for action_ in self.model.actions(parent_node.state)))

                
            else:
                var_dict = state_var_dict[state]
                m.addConstr(
                    gp.quicksum(var for action,var in var_dict.items()) == \
                    gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,state,action_) \
                                for parent_node in self.algo.graph.nodes[state].parents_set for action_ in self.model.actions(parent_node.state)))

               
            

        m.addConstr(
            1 == \
            gp.quicksum(state_var_dict[state_][action_]*self.prob(state_,goal,action_) \
                        for goal in self.goal_list
                        for state_ in self.state_list
                        for action_ in self.model.actions(state_)
                        ), name='const')


        m.addConstr(gp.quicksum(state_var_dict[state][action]*self.secondary_cost(state,action,0) \
                                for state in self.state_list for action in self.model.actions(state)) \
                    <= self.bound[0])



        for state in self.state_list:
            m.addConstr(
                gp.quicksum(delta_state_var_dict[state][action] for action in self.model.actions(state))
                <= 1)

        for goal in self.goal_list:
            m.addConstr(
                gp.quicksum(delta_goal_var_dict[goal][action] for action in self.model.actions(goal))
                <= 1)


        for state in self.state_list:
            for action in self.model.actions(state):
                m.addConstr(
                    state_var_dict[state][action] / X <= delta_state_var_dict[state][action])

        for goal in self.goal_list:
            for action in self.model.actions(goal):
                m.addConstr(
                    goal_var_dict[goal][action] / X <= delta_goal_var_dict[goal][action])


        



        ## Add objective

        m.setObjective(gp.quicksum(state_var_dict[state][action]*self.primary_cost(state,action) \
                                   for state in self.state_list for action in self.model.actions(state)))
        


        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        print('Obj: %g' % m.objVal)

        



    def solve_opt_LP(self):

        m = gp.Model("CSSP")

        X = 100000

        ## Add variables
        state_var_dict = dict()
        goal_var_dict = dict()

        delta_state_var_dict = dict()
        delta_goal_var_dict = dict()

        for state in self.state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(lb=0, name=str((state,action)))

            state_var_dict[state] = dict_temp

        for goal in self.goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(lb=0, name=str((goal,action)))

            goal_var_dict[goal] = dict_temp



        in_var_dict = dict()
        out_var_dict = dict()

        for state in self.state_list:
            in_var_dict[state] = m.addVar(lb=0, name=str(state))
            out_var_dict[state] = m.addVar(lb=0, name=str(state))

        for goal in self.goal_list:
            in_var_dict[goal] = m.addVar(lb=0, name=str(goal))

            
        ## Add constraints

        for state in self.state_list:
            in_var = in_var_dict[state]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,state,action_) \
                                for parent_node in self.algo.graph.nodes[state].parents_set for action_ in self.model.actions(parent_node.state))
                        )


            

            out_var = out_var_dict[state]
            var_dict = state_var_dict[state]
            m.addConstr(out_var == \
                        gp.quicksum(var for action,var in var_dict.items()))
                        
        


        for goal in self.goal_list:
            in_var = in_var_dict[goal]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,goal,action_) \
                                for parent_node in self.algo.graph.nodes[goal].parents_set for action_ in self.model.actions(parent_node.state))
                        )


                

        for state in self.state_list:
            if state==self.init_state:
                m.addConstr(
                    out_var_dict[state] == 1 + in_var_dict[state])

                
            else:
                m.addConstr(
                    out_var_dict[state] == in_var_dict[state])
            

        m.addConstr(
            0.9 <= gp.quicksum(in_var_dict[goal] for goal in self.goal_list))

        m.addConstr(
            1.1 >= gp.quicksum(in_var_dict[goal] for goal in self.goal_list))

        # m.addConstr(
        #     1 == gp.quicksum(in_var_dict[goal] for goal in self.goal_list))





        for i in range(len(self.bound)):

            m.addConstr(gp.quicksum(state_var_dict[state][action]*self.secondary_cost(state,action,i) \
                                    for state in self.state_list for action in self.model.actions(state))
                        <= self.bound[i])




        ## Add objective

        m.setObjective(gp.quicksum(state_var_dict[state][action]*self.primary_cost(state,action) \
                                   for state in self.state_list for action in self.model.actions(state)))     


        m.modelSense = GRB.MINIMIZE
        m.update()
        # m.setParam('method', 1)
        m.optimize()

        print('Obj: %g' % m.objVal)



    def solve_opt_MILP(self, time_limit=3600):

        m = gp.Model("CSSP")

        X = 100000


        ## Add variables
        state_var_dict = dict()
        goal_var_dict = dict()

        delta_state_var_dict = dict()
        delta_goal_var_dict = dict()

        for state in self.state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(lb=0, name=str((state,action)))

            state_var_dict[state] = dict_temp

        for goal in self.goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(lb=0, name=str((goal,action)))

            goal_var_dict[goal] = dict_temp



        for state in self.state_list:
            dict_temp = dict()
            for action in self.model.actions(state):
                dict_temp[action] = m.addVar(vtype=GRB.BINARY)

            delta_state_var_dict[state] = dict_temp

        for goal in self.goal_list:
            dict_temp = dict()
            for action in self.model.actions(goal):
                dict_temp[action] = m.addVar(vtype=GRB.BINARY)

            delta_goal_var_dict[goal] = dict_temp



        in_var_dict = dict()
        out_var_dict = dict()

        for state in self.state_list:
            in_var_dict[state] = m.addVar(lb=0, name=str(state))
            out_var_dict[state] = m.addVar(lb=0, name=str(state))

        for goal in self.goal_list:
            in_var_dict[goal] = m.addVar(lb=0, name=str(goal))

            
        ## Add constraints

        for state in self.state_list:
            in_var = in_var_dict[state]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,state,action_) \
                                for parent_node in self.algo.graph.nodes[state].parents_set for action_ in self.model.actions(parent_node.state))
                        )


            out_var = out_var_dict[state]
            var_dict = state_var_dict[state]
            m.addConstr(out_var == \
                        gp.quicksum(var for action,var in var_dict.items()))
                        
        


        for goal in self.goal_list:
            in_var = in_var_dict[goal]
            m.addConstr(in_var == \
                        gp.quicksum(state_var_dict[parent_node.state][action_]*self.prob(parent_node.state,goal,action_) \
                                for parent_node in self.algo.graph.nodes[goal].parents_set for action_ in self.model.actions(parent_node.state))
                        )

            
                

        for state in self.state_list:
            if state==self.init_state:
                m.addConstr(
                    out_var_dict[state] == 1 + in_var_dict[state])

                
            else:
                m.addConstr(
                    out_var_dict[state] == in_var_dict[state])
            
        m.addConstr(
            0.999 <= gp.quicksum(in_var_dict[goal] for goal in self.goal_list))

        m.addConstr(
            1.001 >= gp.quicksum(in_var_dict[goal] for goal in self.goal_list))


        for i in range(len(self.bound)):

            m.addConstr(gp.quicksum(state_var_dict[state][action]*self.secondary_cost(state,action,i) \
                                    for state in self.state_list for action in self.model.actions(state))
                        <= self.bound[i])

        

        for state in self.state_list:
            m.addConstr(
                gp.quicksum(delta_state_var_dict[state][action] for action in self.model.actions(state))
                <= 1)

        for goal in self.goal_list:
            m.addConstr(
                gp.quicksum(delta_goal_var_dict[goal][action] for action in self.model.actions(goal))
                <= 1)


        for state in self.state_list:
            for action in self.model.actions(state):
                m.addConstr(
                    state_var_dict[state][action] / X <= delta_state_var_dict[state][action])

        for goal in self.goal_list:
            for action in self.model.actions(goal):
                m.addConstr(
                    goal_var_dict[goal][action] / X <= delta_goal_var_dict[goal][action])


        

        ## Add objective

        m.setObjective(gp.quicksum(state_var_dict[state][action]*self.primary_cost(state,action) \
                                   for state in self.state_list for action in self.model.actions(state)))     


        m.Params.TimeLimit = time_limit

        m.modelSense = GRB.MINIMIZE
        m.update()
        m.optimize()

        print('Obj: %g' % m.objVal)

        return m.objVal
        


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
