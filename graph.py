#!/usr/bin/env python

import sys
import string
import random
import time
from collections import deque


class GraphElement(object):
    """
    Generic graph element with a name and a unique ID.
    """

    def __init__(self, name=None):
        self.name = name

    __hash__ = object.__hash__


    def __eq__(self, y):
        return isinstance(self, GraphElement) and isinstance(y, GraphElement) and (self.name == y.name)

    def __ne__(self, other):
        return not self == other


class Node(GraphElement):
    """
    Class for nodes in the graph.
    """

    def __init__(self, state, value_1, value_2, value_3, value_4, value_5, value_6,value_7,value_8,value_9, best_action=None, terminal=False):
        super(Node, self).__init__(state)
        
        self.value_1 = value_1
        self.value_2 = value_2
        self.value_3 = value_3
        self.value_4 = value_4
        self.value_5 = value_5
        self.value_6 = value_6
        self.value_7 = value_7
        self.value_8 = value_8
        self.value_9 = value_9
        self.terminal = terminal  # Terminal flag
        self.state = state
        self.best_action = best_action  # Best action at the node

        self.hide_actions = []

        self.parents_set = set()   # set of all parents in the graph
        self.best_parents_set = set()    # set of parents in the best partial explicit graph
        self.children = dict()     # children dictionary, where key is an action, value is a set of children nodes with probabilities (nested list, e.g., [[node1,prob1], [node2,prob2]])

        self.color = 'w'    # can be 'w':white, 'g':gray, 'b':black, for tricolor implementation of dfs. 


    def set_terminal(self):

        self.terminal = True
        self.value_1 = 0
        self.value_2 = 0
        self.value_3 = 0
        self.value_4 = 0
        self.valeu_5 = 0
        self.value_6 = 0
        self.value_7 = 0
        self.value_8 = 0
        self.valeu_9 = 0





class Graph(GraphElement):
    """
    Class representing an graph.
    """

    def __init__(self, name=None):
        super(Graph, self).__init__(name)
        # Dictionary of nodes mapping their string names to themselves
        self.nodes = {}
        self.root = None
        # Dictionary of operators mapping their string names to themselves
    #     self.operators = {}
    #     # Nested dictionary {parent_key: {operator_key: successors}}
    #     self.hyperedges = {}
    #     # Dictionary from child to sets of parents {child_key: set(parents)}
    #     self.parents = {}

    #     if (sys.version_info > (3, 0)):
    #         # Python 3 code in this block
    #         self.python3 = True
    #     else:
    #         # Python 2 code in this block
    #         self.python3 = False

    # def reset_likelihoods(self):
    #     if self.python3:
    #         for name, node in self.nodes.items():
    #             node.likelihood = 0.0
    #     else:
    #         for name, node in self.nodes.iteritems():
    #             node.likelihood = 0.0

    # # def mark_all_node_unreachable(self):

    # def update_root_and_purge(self, new_root):

    #     # reset all reachable markings in the graph
    #     for name, node in self.nodes.items():
    #         node.reachable = False

    #     queue = deque([new_root])
    #     marked = set()
    #     marked.add(new_root)
    #     new_root.reachable = True

    #     while len(queue) > 0:
    #         node = queue.popleft()

    #         if not node.terminal:  # node is not terminal

    #             children = self.all_descendants(node)

    #             for c in children:
    #                 if c not in marked:
    #                     c.reachable = True
    #                     marked.add(c)
    #                     queue.append(c)
    #         # else:  # no best action has been assigned yet
    #             # should not need to do anything because we are marking
    #             # children of nodes with best_actions

    #     for name, node in self.nodes.items():
    #         if not node.reachable:
    #             del self.nodes[name]

    # def set_nodes(self, new_nodes):
    #     self.nodes = new_nodes

    # def set_operators(self, new_operators):
    #     self.operators = new_operators

    # def set_hyperedges(self, new_hyperedges):
    #     if isinstance(new_hyperedges, dict):
    #         self.hyperedge_dict = new_hyperedges
    #     else:
    #         raise TypeError('Hyperedges should be given in dictionary form')

    # def set_parents(self, new_parents):
    #     if isinstance(new_parents, dict):
    #         self.parent_dict = new_parents
    #     else:
    #         raise TypeError('Node parents should be given in dictionary form')

    # def set_root(self, new_root):
    #     if isinstance(new_root, RAOStarGraphNode):
    #         self.root = new_root
    #         self.add_node(self.root)
    #     else:
    #         raise TypeError(
    #             'The root of the hypergraph must be of type RAOStarGraphNode.')


    def add_root(self, state, value_1=None, value_2=None, value_3=None,value_4=None,value_5=None,value_6=None, value_7=None,value_8=None,value_9=None, best_action=None, terminal=False):
        """Adds a node to the hypergraph."""
        if not state in self.nodes:
            self.nodes[state] = Node(state, value_1, value_2, value_3, value_4, value_5,value_6, value_7, value_8, value_9, best_action, terminal)
            self.root = self.nodes[state]
            return True
        else:
            return False

    
    def add_node(self, state, value_1=None, value_2=None, value_3=None,value_4=None,value_5=None,value_6=None, value_7=None,value_8=None,value_9=None, best_action=None, terminal=False):
        """Adds a node to the hypergraph."""
        if not state in self.nodes:
            self.nodes[state] = Node(state, value_1, value_2, value_3, value_4, value_5,value_6, value_7, value_8, value_9, best_action, terminal)
            return True
        else:
            return False

    # def add_operator(self, op):
    #     """Adds an operators to the hypergraph."""
    #     if not op in self.operators:
    #         self.operators[op] = op

    # def add_hyperedge(self, parent_obj, child_obj_list, prob_list, prob_safe_list, op_obj):
    #     """Adds a hyperedge between a parent and a list of child nodes, adding
    #     the nodes to the graph if necessary."""
        
    #     # Makes sure all nodes and operator are part of the graph.
    #     self.add_node(parent_obj)    
    #     self.add_operator(op_obj)


    #     for child in child_obj_list:
    #         self.add_node(child)

    #     # Adds the hyperedge
    #     # TODO: check if the hashing is being done correctly here by __hash__
    #     if parent_obj in self.hyperedges:

    #         # #TODO:
    #         # #Symptom: operators at the hypergraph nodes where being duplicated,
    #         # #even though actions the hypergraph models (the operator names),
    #         # #were not.
    #         # #
    #         # #Debug conclusions: operators were using their memory ids as hash
    #         # #keys, causing operators with the same action (name) to be considered
    #         # #different objects. The duplication would manifest itself when a node
    #         # #already with outgoing hyperedges (parent_obj in self.hyperedges) was
    #         # #later dequed and given the same operator. Fortunately, the tests
    #         # #revealed that different operators with the same action would yield
    #         # #the same set of children nodes, which indicates that the expansion
    #         # #is correctly implemented.
    #         #
    #         # if op_obj in self.hyperedges[parent_obj]:
    #         #     #TODO: this should be removed, once I'm confident that the algorithm
    #         #     #is handling the hypergraph correctly. It checks whether the two
    #         #     #copies of the same operator yielded the same children at the same
    #         #     #parent node (which is a requirement), and opens a debugger if they
    #         #     #don't
    #         #     prev_children = self.hyperedges[parent_obj][op_obj]
    #         #     if len(prev_children)!=len(child_obj_list):
    #         #         print('WARNING: operator %s at node %s yielded children lists with different lengths'%(op_obj.name,parent_obj.name))
    #         #         import ipdb; ipdb.set_trace()
    #         #         pass
    #         #
    #         #     for child in child_obj_list:
    #         #         if not child in prev_children:
    #         #             print('WARNING: operator %s at node %s yielded different sets of children'%(op_obj.name,parent_obj.name))
    #         #             import ipdb; ipdb.set_trace()
    #         #             pass

    #         self.hyperedges[parent_obj][op_obj] = child_obj_list
    #     else:
    #         self.hyperedges[parent_obj] = {op_obj: child_obj_list}

    #     # Records the mapping from children to parent nodes
    #     for i, child in enumerate(child_obj_list):
    #         if not (child.unique_id in self.parents):
    #             self.parents[child.unique_id] = set()
    #         # Added association of action and probability to each parent
    #         # This is so we can match transition probabilities when calculating
    #         # likelihoods in the policy
    #         self.parents[child.unique_id].add((parent_obj, op_obj, prob_list[i], prob_safe_list[i]))
    #         # if len(self.parents)>1:
    #         #     print(self.parents)
    #         #     time.sleep(1000)

    # def remove_all_hyperedges(self, node):
    #     """Removes all hyperedges at a node."""
    #     if node in self.hyperedges:
    #         del self.hyperedges[node]

    # def all_node_operators(self, node):
    #     """List of all operators at a node."""
    #     return list(self.hyperedges[node].keys()) if node in self.hyperedges else []

    # def all_descendants(self, node):
    #     """List of all descendants of a node, from all hyperedges"""
    #     '''Currently includes repititions!'''
    #     operators = self.all_node_operators(node)
    #     descendants = []
    #     for o in operators:
    #         descendants.extend(self.hyperedge_successors(node, o))
    #     return descendants

    # def all_node_ancestors(self, node):
    #     """Set of all node parents, considering all hyperedges."""
    #     if node.unique_id in self.parents:
    #         return self.parents[node.unique_id]
    #     return set()

    # def policy_successors(self, node):
    #     if node.best_action:
    #         return self.hyperedge_successors(node, node.best_action)
    #     return []

    # def hyperedge_successors(self, node, act):
    #     """List of children associated to a hyperedge."""
    #     if node in self.hyperedges and act in self.hyperedges[node]:
    #         return self.hyperedges[node][act]
    #     return []

    # def has_node(self, node):
    #     """Returns whether the hypergraph contains the node"""
    #     return (node.name in self.nodes)

    # def has_operator(self, op):
    #     """Returns whether the hypergraph contains the operator."""
    #     return (op in self.operators)

    # def has_ancestor(self, node):
    #     """Whether a node has ancestors in the graph."""
    #     return (node.unique_id in self.parents)
