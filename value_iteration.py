#!/usr/bin/env python

import sys
import time
from utils import *
from graph import Node, Graph


class VI(object):

    def __init__(self, model, constrained=False, VI_epsilon=1e-50, VI_max_iter=100000,bounds=[], alpha=[]):

        self.model = model
        self.constrained=constrained
        
        if not self.constrained:
            self.value_num = 1
        else:
            self.value_num = len(alpha)+1

            if self.value_num > 3:
                raise ValueError("more than 2 constraints is not implemented yet.")
        
        self.VI_epsilon = VI_epsilon
        self.VI_max_iter = VI_max_iter
        
        
        self.graph = Graph(name='G')
        if not self.constrained:
            self.graph.add_root(model.init_state, self.model.heuristic(model.init_state))
        else:
            self.graph.add_root(model.init_state, *self.model.heuristic(model.init_state))



        self.alpha = alpha
        self.bounds = bounds
        self.debug_k = 0



    def solve(self):

        self.expand_all()
        self.value_iteration(list(self.graph.nodes.values()))
        self.update_best_partial_graph()
        return self.extract_policy()
    

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



    def expand(self,expanded_node):

        state = expanded_node.state
        actions = self.model.actions(state)

        children_nodes = []
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

                children_nodes.append(child)

            expanded_node.children[action] = children_list


        return children_nodes



    def compute_value(self,node,action):

        if self.value_num == 1:
            value_1 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
           
            return value_1, None, None

        elif self.value_num == 2:
            value_1, value_2 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                
            return value_1, value_2, None

        elif self.value_num ==3:
            value_1, value_2, value_3 = self.model.cost(node.state,action)
            for child, child_prob in node.children[action]:
                value_1 = value_1 + child.value_1*child_prob
                value_2 = value_2 + child.value_2*child_prob
                value_3 = value_3 + child.value_3*child_prob
                
            return value_1, value_2, value_3






    def value_iteration(self, Z, epsilon=1e-50, max_iter=100000,return_on_policy_change=False):

        
        iter=0

        V_prev = dict()
        V_new = dict()
        
        max_error = 10**10
        while not max_error < epsilon:
            max_error = -1
            for node in Z:
                if node.terminal==False:
                    
                    if not self.constrained:
                        V_prev[node.state] = node.value_1
                    else:
                        V_prev[node.state] = self.compute_weighted_value(node.value_1,node.value_2,node.value_3)

 
                    actions = self.model.actions(node.state)
                        
                    
                    if not self.constrained:
                        min_value = float('inf')
                    else:
                        min_value = [float('inf')]*self.value_num
                    weighted_min_value = float('inf')

                    prev_best_action = node.best_action
                    best_action = None

                    for action in actions:
                        
                        new_value_1, new_value_2, new_value_3 = self.compute_value(node,action)

                        if self.constrained==False:  # simple SSP case
                            if new_value_1 < min_value:
                                min_value = new_value_1
                                best_action = action

                        else:
                            weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3)

                            if weighted_value < weighted_min_value:
                                min_value_1 = new_value_1
                                min_value_2 = new_value_2
                                min_value_3 = new_value_3
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

                    node.best_action = best_action

                    error = abs(V_prev[node.state] - V_new[node.state])
                    if error > max_error:
                        max_error = error


                    if return_on_policy_change==True:
                        if prev_best_action != best_action:
                            return False


            iter += 1
                    
            if iter > max_iter:
                print("Maximun number of iteration reached.")
                break

        return V_new
        


        


    # def value_iteration(self, Z, epsilon=1e-50, max_iter=100000,return_on_policy_change=False):

    #     # if self.debug_k==15:
    #     #     for z in Z:
    #     #         print(z.state)
    #     #     print(len(Z))

    #     iter=0

    #     V_prev = dict()
    #     V_new = dict()
    #     V_new_1 = dict()
    #     V_new_2 = dict()
    #     V_new_3 = dict()
        



    #     max_error = 10**10
    #     while not max_error < epsilon:
    #         max_error = -1
    #         for node in Z:
    #             if node.terminal==False:
                    
    #                 if not self.constrained:
    #                     V_prev[node.state] = node.value_1
    #                 else:
    #                     V_prev[node.state] = self.compute_weighted_value(node.value_1,node.value_2,node.value_3)

    #                 actions = self.model.actions(node.state)
    #                 if not self.constrained:
    #                     min_value = float('inf')
    #                 else:
    #                     min_value = [float('inf')]*self.value_num
    #                 weighted_min_value = float('inf')

    #                 prev_best_action = node.best_action
    #                 best_action = None

    #                 for action in actions:

    #                     new_value_1, new_value_2, new_value_3 = self.compute_value(node,action)

    #                     if self.constrained==False:  # simple SSP case
    #                         if new_value_1 < min_value:
    #                             min_value = new_value_1
    #                             best_action = action

    #                     else:
    #                         if self.Lagrangian==False:
    #                             raise ValueError("need to be implemented for constrained case.")
    #                         else:
    #                             weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3)

    #                             if weighted_value < weighted_min_value:
    #                                 min_value_1 = new_value_1
    #                                 min_value_2 = new_value_2
    #                                 min_value_3 = new_value_3
    #                                 weighted_min_value = weighted_value
    #                                 best_action = action

    #                 V_new[node.state] = min_value
    #                 if self.constrained:
    #                     V_new_1[node.state] = min_value_1
    #                     V_new_2[node.state] = min_value_2
    #                     V_new_3[node.state] = min_value_3

    #                 error = abs(V_prev[node.state] - V_new[node.state])
    #                 if error > max_error:
    #                     max_error = error
                    
    #                 if return_on_policy_change==True:
    #                     if prev_best_action != best_action:
    #                         return False


    #         for node in Z:
    #             if node.terminal==False:
    #                 if not self.constrained:
    #                     node.value_1 = V_new[node.state]
    #                 else:
    #                     node.value_1 = V_new_1[node.state]
    #                     node.value_2 = V_new_2[node.state]
    #                     node.value_3 = V_new_3[node.state]

    #         iter += 1
                    
    #         if iter > max_iter:
    #             print("Maximun number of iteration reached.")
    #             break

    #     return V_new
    





    def update_best_partial_graph(self, Z=None, V_new=None):

        for state,node in self.graph.nodes.items():
            node.best_parents_set = set()
        
        visited = set()
        queue = set([self.graph.root])
        self.graph.root.color = 'w'

        while queue:

            node = queue.pop()

            if node in visited:
                continue

            else:
                if node.children!=dict():

                    actions = self.model.actions(node.state)
                    if not self.constrained:
                        min_value = float('inf')
                    else:
                        min_value = [float('inf')]*self.value_num
                    weighted_min_value = float('inf')
                    
                    for action in actions:
                        new_value_1,new_value_2,new_value_3 = self.compute_value(node,action)

                        if self.constrained==False:
                            if new_value_1 < min_value:
                                node.best_action = action
                                min_value = new_value_1

                        else:
                            weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3)

                            if weighted_value < weighted_min_value:
                                min_value_1 = new_value_1
                                min_value_2 = new_value_2
                                min_value_3 = new_value_3
                                weighted_min_value = weighted_value
                                best_action = action
                                    

                    if not self.constrained:
                        node.value_1 = min_value
                    else:
                        node.value_1 = min_value_1
                        node.value_2 = min_value_2
                        node.value_3 = min_value_3
                        
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)
                        child.best_parents_set.add(node)
                        child.color = 'w'

            visited.add(node)
    

        

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
        




    def compute_weighted_value(self,value_1, value_2, value_3):

        if self.value_num == 1:
            raise ValueError("seems there is no constraint but weighted value is being computed.")

        elif self.value_num == 2:
            weighted_cost = value_1 + self.alpha[0]*(value_2 - self.bounds[0])

        elif self.value_num == 3:
            weighted_cost = value_1 + self.alpha[0]*value_2 + self.alpha[1]*value_3

        return weighted_cost
    


    
    def policy_evaluation(self, policy, epsilon=1e-50):

        V_prev = dict()
        V_new = dict()
        
        max_error = 10**10
        while not max_error < epsilon:
            max_error = -1

            for state, node in self.graph.nodes.items():
                if node.terminal==False:
                    
                    if not self.constrained:
                        V_prev[node.state] = node.value_1
                    else:
                        V_prev[node.state] = self.compute_weighted_value(node.value_1,node.value_2,node.value_3)

 
                    
                    if not self.constrained:
                        min_value = float('inf')
                    else:
                        min_value = [float('inf')]*self.value_num
                        
                    weighted_min_value = float('inf')

                        
                    new_value_1, new_value_2, new_value_3 = self.compute_value(node,policy[node.state])

                    if self.constrained==False:  # simple SSP case
                        min_value = new_value_1
                        

                    else:
                        weighted_value = self.compute_weighted_value(new_value_1, new_value_2, new_value_3)

                        min_value_1 = new_value_1
                        min_value_2 = new_value_2
                        min_value_3 = new_value_3
                        weighted_min_value = weighted_value


                    if not self.constrained:
                        V_new[node.state] = min_value
                        node.value_1 = min_value
                    else:
                        V_new[node.state] = weighted_min_value
                        node.value_1 = min_value_1
                        node.value_2 = min_value_2
                        node.value_3 = min_value_3


                    error = abs(V_prev[node.state] - V_new[node.state])
                    if error > max_error:
                        max_error = error
