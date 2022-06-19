#!/usr/bin/env python

import sys
import time
from functools import reduce
from utils import *
from graph import Node, Graph
from collections import deque


class ILAOStar(object):

    def __init__(self, model, constrained=False, method='VI', VI_epsilon=1e-50, VI_max_iter=100000, convergence_epsilon=1e-50, \
                 bounds=[], alpha=[], Lagrangian=False, incremental=False, tweaking_node=None, head=set(), blocked_action_set=set()):

        self.model = model
        self.constrained = constrained

        if not self.constrained:
            self.value_num = 1
        else:
            self.value_num = len(alpha)+1

        self.method = method
        self.VI_epsilon = VI_epsilon
        self.VI_max_iter = VI_max_iter
        self.convergence_epsilon = convergence_epsilon
        
        self.bounds = bounds
        self.alpha = alpha
        self.Lagrangian=Lagrangian
        
        self.graph = Graph(name='G')
        if not self.constrained:
            self.graph.add_root(model.init_state, self.model.heuristic(model.init_state))
        else:
            self.graph.add_root(model.init_state, *self.model.heuristic(model.init_state))

        self.fringe = {self.graph.root}


        ## for second stage (incremental update)

        self.incremental = incremental
        self.tweaking_node = tweaking_node
        self.head = head
        self.blocked_action_set = blocked_action_set

        self.debug_k = 0

    def solve(self):

        while not self.is_termination():

            self.dfs_update(self.graph.root)
            self.update_best_partial_graph()  ## this ensures that best partial graph nodes are initialized with "w" color, for next dfs update. 
            self.update_fringe()

            self.debug_k += 1

        return self.extract_policy()


    def dfs_update(self,node):
        # print("hi")

        node.color = 'g'

        if node.children != dict():
            children = node.children[node.best_action]

            for child,child_prob in children:
               if child.color == 'w':
                    self.dfs_update(child)
        

        if not self.model.is_terminal(node.state) and node.best_action==None:  ## if this node has not been expanded and not terminal, then expand.
            self.expand(node)

            
        if not self.model.is_terminal(node.state):   ## if this node is not terminal, evaluate.

            if self.incremental==False:
                actions = self.model.actions(node.state)
            else:
                actions = self.incremental_update_action_model(node)
            
            if self.value_num == 1:
                min_value = float('inf')
            else:
                min_value = [float('inf')]*self.value_num
                
            weighted_min_value = float('inf')    

            for action in actions:
                new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9 = self.compute_value(node,action)

                if self.constrained==False:
                    if new_value_1 < min_value:
                        node.best_action = action
                        min_value = new_value_1

                else:
                    if self.Lagrangian==False:
                        raise ValueError("need to be implemented for constrained case.")
                    else:
                        weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9)

                        if weighted_value < weighted_min_value:
                            node.best_action = action
                            min_value_1 = new_value_1
                            min_value_2 = new_value_2
                            min_value_3 = new_value_3
                            min_value_4 = new_value_4
                            min_value_5 = new_value_5
                            min_value_6 = new_value_6
                            min_value_7 = new_value_7
                            min_value_8 = new_value_8
                            min_value_9 = new_value_9
                            weighted_min_value = weighted_value

            if not self.constrained:
                node.value_1 = min_value
            else:
                node.value_1 = min_value_1
                node.value_2 = min_value_2
                node.value_3 = min_value_3
                node.value_4 = min_value_4
                node.value_5 = min_value_5
                node.value_6 = min_value_6
                node.value_7 = min_value_7
                node.value_8 = min_value_8
                node.value_9 = min_value_9
            children = node.children[node.best_action]
            

    
    def is_termination(self):

        if self.fringe:
            return False
        else:
            if self.method=='VI':
                if self.convergence_test():
                    return True
                else:
                    
                    while True:                        

                        
                        self.update_best_partial_graph(None,None)

                        self.update_fringe()

                        if self.fringe:
                            return False

                        if self.convergence_test():
                            return True

            return True



    def expand(self,expanded_node=None):
        if expanded_node==None:
            expanded_node = self.fringe.pop()
            
        state = expanded_node.state

        actions = self.model.actions(state)

        for action in actions:
            children = self.model.state_transitions(state,action)

            children_list = []
            for child_state, child_prob in children:
                if child_state in self.graph.nodes:
                    child = self.graph.nodes[child_state]
                else:
                    if not self.constrained:
                        self.graph.add_node(child_state, self.model.heuristic(child_state))
                    else:
                        self.graph.add_node(child_state, *self.model.heuristic(child_state))
                    child = self.graph.nodes[child_state]

                    if self.model.is_terminal(child.state):
                        child.set_terminal()
                
                child.parents_set.add(expanded_node)
                children_list.append([child,child_prob])

            expanded_node.children[action] = children_list

        return expanded_node



    def convergence_test(self):

        Z = self.get_best_solution_nodes()

        if self.incremental == False:
            return self.value_iteration(Z, epsilon=self.convergence_epsilon, return_on_policy_change=True)

        else:
            return self.prioritized_value_iteration(Z, epsilon=self.convergence_epsilon, return_on_policy_change=True)


    def get_best_solution_nodes(self):
        policy = self.extract_policy()
        Z = []
        for state in list(policy.keys()):
            Z.append(self.graph.nodes[state])

        return Z
            

    def temp(self, child):
        return child[0].value_1 * child[1]
    
    def compute_value(self,node,action):
        
        if self.value_num == 1:
            value_1 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
           
            return value_1, None, None, None, None, None, None, None, None

        elif self.value_num == 2:
            value_1, value_2 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                
            return value_1, value_2, None, None, None, None, None, None, None

        elif self.value_num ==3:
            value_1, value_2, value_3 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                
            return value_1, value_2, value_3, None, None, None, None, None, None

        
        elif self.value_num ==4:
            value_1, value_2, value_3, value_4 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                value_4 = value_4 + child.value_4*child_prob
                
            return value_1, value_2, value_3, value_4, None, None, None, None, None


        elif self.value_num ==5:
            value_1, value_2, value_3, value_4, value_5 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                value_4 = value_4 + child.value_4*child_prob
                value_5 = value_5 + child.value_5*child_prob
                
            return value_1, value_2, value_3, value_4, value_5, None, None, None, None

        elif self.value_num ==6:
            value_1, value_2, value_3, value_4, value_5, value_6 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                value_4 = value_4 + child.value_4*child_prob
                value_5 = value_5 + child.value_5*child_prob
                value_6 = value_6 + child.value_6*child_prob
                
            return value_1, value_2, value_3, value_4, value_5, value_6, None, None, None

        elif self.value_num ==7:
            value_1, value_2, value_3, value_4, value_5, value_6, value_7 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                value_4 = value_4 + child.value_4*child_prob
                value_5 = value_5 + child.value_5*child_prob
                value_6 = value_6 + child.value_6*child_prob
                value_7 = value_7 + child.value_7*child_prob
                
                
            return value_1, value_2, value_3, value_4, value_5, value_6, value_7, None, None

        elif self.value_num ==8:
            value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                value_4 = value_4 + child.value_4*child_prob
                value_5 = value_5 + child.value_5*child_prob
                value_6 = value_6 + child.value_6*child_prob
                value_7 = value_7 + child.value_7*child_prob
                value_8 = value_8 + child.value_8*child_prob
                
            return value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, None

        elif self.value_num ==9:
            value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                value_4 = value_4 + child.value_4*child_prob
                value_5 = value_5 + child.value_5*child_prob
                value_6 = value_6 + child.value_6*child_prob
                value_7 = value_7 + child.value_7*child_prob
                value_8 = value_8 + child.value_8*child_prob
                value_9 = value_9 + child.value_9*child_prob
                
            return value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9



    def compute_weighted_value(self,value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9):

        if self.value_num == 1:
            raise ValueError("seems there is no constraint but weighted value is being computed.")

        elif self.value_num == 2:
            weighted_cost = value_1 + self.alpha[0]*(value_2 - self.bounds[0])

        elif self.value_num == 3:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3

        elif self.value_num == 4:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3 + self.alpha[2]*value_4

        elif self.value_num == 5:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3 + self.alpha[2]*value_4 + self.alpha[3]*value_5

        elif self.value_num == 6:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3 + self.alpha[2]*value_4 + self.alpha[3]*value_5 + self.alpha[4]*value_6

        elif self.value_num == 7:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3 + self.alpha[2]*value_4 + self.alpha[3]*value_5 + self.alpha[4]*value_6 + self.alpha[5]*value_7

        elif self.value_num == 8:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3 + self.alpha[2]*value_4 + self.alpha[3]*value_5 + self.alpha[4]*value_6 + self.alpha[5]*value_7 \
                             + self.alpha[6]*value_8

        elif self.value_num == 9:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3 + self.alpha[2]*value_4 + self.alpha[3]*value_5 + self.alpha[4]*value_6 + self.alpha[5]*value_7 \
                             + self.alpha[6]*value_8 + self.alpha[7]*value_9

        return weighted_cost
    

    def value_iteration(self, Z, epsilon=1e-50, max_iter=100000,return_on_policy_change=False):

        iter=0

        V_prev = dict()
        V_new = dict()
        
        max_error = 10**10
        while not max_error < epsilon:
            max_error = -1
            for node in Z:
                if node.terminal==False and node.best_action!=None:
                    
                    if not self.constrained:
                        V_prev[node.state] = node.value_1
                    else:
                        V_prev[node.state] = self.compute_weighted_value(node.value_1,node.value_2,node.value_3,node.value_4,node.value_5, \
                                                                         node.value_6,node.value_7,node.value_8,node.value_9)

 
                    if self.incremental==False:
                        actions = self.model.actions(node.state)
                    else:
                        actions = self.incremental_update_action_model(node)                       
                        
                    
                    if not self.constrained:
                        min_value = float('inf')
                    else:
                        min_value = [float('inf')]*self.value_num
                    weighted_min_value = float('inf')

                    prev_best_action = node.best_action
                    best_action = None

                    for action in actions:
                        
                        new_value_1, new_value_2, new_value_3, new_value_4, new_value_5,new_value_6, new_value_7, new_value_8, new_value_9 = self.compute_value(node,action)

                        if self.constrained==False:  # simple SSP case
                            if new_value_1 < min_value:
                                min_value = new_value_1
                                best_action = action

                        else:
                            if self.Lagrangian==False:
                                raise ValueError("need to be implemented for constrained case.")
                            else:
                                weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3, new_value_4, new_value_5,new_value_6, new_value_7, new_value_8, new_value_9)

                                if weighted_value < weighted_min_value:
                                    min_value_1 = new_value_1
                                    min_value_2 = new_value_2
                                    min_value_3 = new_value_3
                                    min_value_4 = new_value_4
                                    min_value_5 = new_value_5
                                    min_value_6 = new_value_6
                                    min_value_7 = new_value_7
                                    min_value_8 = new_value_8
                                    min_value_9 = new_value_9
                                    weighted_min_value = weighted_value
                                    best_action = action


                    if not self.constrained:
                        V_new[node.state] = min_value
                        node.value_1 = min_value
                    else:
                        V_new[node.state] = weighted_min_value
                        node.value_1 = min_value_1
                        node.value_2 = min_value_2
                        node.value_3 = min_value_3
                        node.value_4 = min_value_4
                        node.value_5 = min_value_5
                        node.value_6 = min_value_6
                        node.value_7 = min_value_7
                        node.value_8 = min_value_8
                        node.value_9 = min_value_9

                    node.best_action = best_action


                    error = abs(V_prev[node.state] - V_new[node.state])
                    if error > max_error:
                        max_error = error


                    if return_on_policy_change==True:
                        if prev_best_action != best_action:
                            return False


            iter += 1
                    
            if iter > max_iter:
                break

        return V_new

    def prioritized_value_iteration(self, Z, epsilon=1e-50, max_iter=100000,return_on_policy_change=False):

        iter=0

        V_prev = dict()
        V_new = dict()
        
        max_error = 10**10
        while not max_error < epsilon:
            max_error = -1

            queue = set([self.tweaking_node])
            visited = set()

            while queue:

                next_parents = set()

                while queue:

                    node = queue.pop()
                    visited.add(node)

                    if node.terminal==False and node.best_action!=None:

                        if not self.constrained:
                            V_prev[node.state] = node.value_1
                        else:
                            V_prev[node.state] = self.compute_weighted_value(node.value_1,node.value_2,node.value_3,node.value_4,node.value_5,\
                                                                             node.value_6,node.value_7,node.value_8,node.value_9)


                        if self.incremental==False:
                            actions = self.model.actions(node.state)
                        else:
                            actions = self.incremental_update_action_model(node)                       


                        if not self.constrained:
                            min_value = float('inf')
                        else:
                            min_value = [float('inf')]*self.value_num
                        weighted_min_value = float('inf')

                        prev_best_action = node.best_action
                        best_action = None

                        for action in actions:

                            new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9 = self.compute_value(node,action)

                            if self.constrained==False:  # simple SSP case
                                if new_value_1 < min_value:
                                    min_value = new_value_1
                                    best_action = action

                            else:
                                if self.Lagrangian==False:
                                    raise ValueError("need to be implemented for constrained case.")
                                else:
                                    weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9)

                                    if weighted_value < weighted_min_value:
                                        min_value_1 = new_value_1
                                        min_value_2 = new_value_2
                                        min_value_3 = new_value_3
                                        min_value_4 = new_value_4
                                        min_value_5 = new_value_5
                                        min_value_6 = new_value_6
                                        min_value_7 = new_value_7
                                        min_value_8 = new_value_8
                                        min_value_9 = new_value_9
                                        weighted_min_value = weighted_value
                                        best_action = action


                        if not self.constrained:
                            V_new[node.state] = min_value
                            node.value_1 = min_value
                        else:
                            V_new[node.state] = weighted_min_value
                            node.value_1 = min_value_1
                            node.value_2 = min_value_2
                            node.value_3 = min_value_3
                            node.value_4 = min_value_4
                            node.value_5 = min_value_5
                            node.value_6 = min_value_6
                            node.value_7 = min_value_7
                            node.value_8 = min_value_8
                            node.value_9 = min_value_9

                        node.best_action = best_action


                        error = abs(V_prev[node.state] - V_new[node.state])
                        if error > max_error:
                            max_error = error


                        if return_on_policy_change==True:
                            if prev_best_action != best_action:
                                return False


                    parents = node.best_parents_set

                    for parents_node in parents:
                        if parents_node not in visited:
                            next_parents.add(parents_node)

                queue = next_parents
                    

            iter += 1
                    
            if iter > max_iter:
                break

        return V_new
    

    def update_best_partial_graph(self, Z=None, V_new=None):

        for state,node in self.graph.nodes.items():
            node.best_parents_set = set()
        
        visited = set()
        queue = deque([self.graph.root])
        self.graph.root.color = 'w'

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                if node.children!=dict():

                    
                    if self.incremental==False:
                        actions = self.model.actions(node.state)
                    else:
                        actions = self.incremental_update_action_model(node)

                    if not self.constrained:
                        min_value = float('inf')
                    else:
                        min_value = [float('inf')]*self.value_num
                    weighted_min_value = float('inf')
                    
                    for action in actions:
                        new_value_1,new_value_2,new_value_3,new_value_4,new_value_5,new_value_6,new_value_7,new_value_8,new_value_9 = self.compute_value(node,action)

                        if self.constrained==False:
                            if new_value_1 < min_value:
                                node.best_action = action
                                min_value = new_value_1

                        else:
                            if self.Lagrangian==False:
                                raise ValueError("need to be implemented for constrained case.")
                            else:
                                weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9)

                                if weighted_value < weighted_min_value:
                                    node.best_action = action
                                    min_value_1 = new_value_1
                                    min_value_2 = new_value_2
                                    min_value_3 = new_value_3
                                    min_value_4 = new_value_4
                                    min_value_5 = new_value_5
                                    min_value_6 = new_value_6
                                    min_value_7 = new_value_7
                                    min_value_8 = new_value_8
                                    min_value_9 = new_value_9
                                    weighted_min_value = weighted_value
                                    

                    if not self.constrained:
                        node.value_1 = min_value
                    else:
                        node.value_1 = min_value_1
                        node.value_2 = min_value_2
                        node.value_3 = min_value_3
                        node.value_4 = min_value_4
                        node.value_5 = min_value_5
                        node.value_6 = min_value_6
                        node.value_7 = min_value_7
                        node.value_8 = min_value_8
                        node.value_9 = min_value_9
                        
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.append(child)
                        child.best_parents_set.add(node)
                        child.color = 'w'

            visited.add(node)



    def update_fringe(self):

        fringe = set()
        queue = deque([self.graph.root])
        visited = set()

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                if node.best_action!=None:  # if this node has been expanded
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.append(child)

                elif node.terminal != True:
                    fringe.add(node)

            visited.add(node)

        self.fringe = fringe        

        
    def extract_policy(self):

        queue = set([self.graph.root])
        policy = dict()

        while queue:

            node = queue.pop()

            if node.state in policy:
                continue

            else:
                if node.best_action!=None:
                    policy[node.state] = node.best_action
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)

                elif node.terminal==True:
                    policy[node.state] = "Terminal"

                else:
                    raise ValueError("Best partial graph has non-expanded fringe node.")

        return policy


    def print_policy(self):

        policy = self.extract_policy()

        for state, action in policy.items():
            print(state, ' : ', action, self.graph.nodes[state].value,self.graph.nodes[state].terminal )



    def incremental_update_action_model(self,node):
        if node in self.head:
            actions = [node.best_action]
            
        elif node==self.tweaking_node:
            
            actions = self.model.actions(node.state)[:]
            for blocked_action in self.blocked_action_set:
                actions.remove(blocked_action)

        else:
            actions = self.model.actions(node.state)

        return actions



    def get_values(self,node):

        if self.value_num == 1:
            return node.value_1, None, None, None, None, None, None, None, None

        elif self.value_num ==2:
            return node.value_1, node.value_2, None, None, None, None, None, None, None

        elif self.value_num ==3:
            return node.value_1, node.value_2, node.value_3, None, None, None, None, None, None

        elif self.value_num ==4:
            return node.value_1, node.value_2, node.value_3, node.value_4, None, None, None, None, None

        elif self.value_num ==5:
            return node.value_1, node.value_2, node.value_3, node.value_4, node.value_5, None, None, None, None

        elif self.value_num ==6:
            return node.value_1, node.value_2, node.value_3, node.value_4, node.value_5, node.value_6, None, None, None

        elif self.value_num ==7:
            return node.value_1, node.value_2, node.value_3, node.value_4, node.value_5, node.value_6, node.value_7, None, None

        elif self.value_num ==8:
            return node.value_1, node.value_2, node.value_3, node.value_4, node.value_5, node.value_6, node.value_7, node.value_8, None

        elif self.value_num ==9:
            return node.value_1, node.value_2, node.value_3, node.value_4, node.value_5, node.value_6, node.value_7, node.value_8, node.value_9

        








##################################################################################
########################## below are functions for debugging #####################
########################## test all policies to visually inspect #################
##################################################################################



    def expand_all(self):

        visited = set()
        queue = set([self.graph.root])

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                children_nodes = self.expand(node)

                for child in children_nodes:
                    if child.terminal==False and child not in visited:
                        queue.add(child)
                
            visited.add(node)        



    def policy_evaluation(self, policy, epsilon=1e-50):

        V_prev = dict()
        V_new = dict()
        
        max_error = 10**10
        while not max_error < epsilon:
            max_error = -1
            
            for state, node in self.graph.nodes.items():
                if node.terminal==False and state in policy:
                    
                    if not self.constrained:
                        V_prev[node.state] = node.value_1
                    else:
                        V_prev[node.state] = self.compute_weighted_value(node.value_1,node.value_2,node.value_3,node.value_4,node.value_5,\
                                                                         node.value_6,node.value_7,node.value_8,node.value_9)

 
                    
                    if not self.constrained:
                        min_value = float('inf')
                    else:
                        min_value = [float('inf')]*self.value_num
                        
                    weighted_min_value = float('inf')

                        
                    new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9 = self.compute_value(node,policy[node.state])

                    if self.constrained==False:  # simple SSP case
                        min_value = new_value_1
                        

                    else:
                        if self.Lagrangian==False:
                            raise ValueError("need to be implemented for constrained case.")
                        else:
                            weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3, new_value_4, new_value_5, new_value_6, new_value_7, new_value_8, new_value_9)

                            min_value_1 = new_value_1
                            min_value_2 = new_value_2
                            min_value_3 = new_value_3
                            min_value_4 = new_value_4
                            min_value_5 = new_value_5
                            min_value_6 = new_value_6
                            min_value_7 = new_value_7
                            min_value_8 = new_value_8
                            min_value_9 = new_value_9
                            weighted_min_value = weighted_value


                    if not self.constrained:
                        V_new[node.state] = min_value
                        node.value_1 = min_value
                    else:
                        V_new[node.state] = weighted_min_value
                        node.value_1 = min_value_1
                        node.value_2 = min_value_2
                        node.value_3 = min_value_3
                        node.value_4 = min_value_4
                        node.value_5 = min_value_5
                        node.value_6 = min_value_6
                        node.value_7 = min_value_7
                        node.value_8 = min_value_8
                        node.value_9 = min_value_9


                    error = abs(V_prev[node.state] - V_new[node.state])
                    if error > max_error:
                        max_error = error


