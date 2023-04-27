#!/usr/bin/env python

import sys
import time
import math
from utils import *
from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar

import copy
from heapq import *


import numpy as np

class CSSPSolver(object):

    def __init__(self, model, VI_epsilon=1e-50, VI_max_iter=100000, convergence_epsilon=1e-50, resolve_epsilon=1e-5, bounds=[]):

        self.model = model
        self.bounds = bounds
        self.resolve_epsilon = resolve_epsilon

        self.algo = ILAOStar(self.model,constrained=True,VI_epsilon=VI_epsilon, convergence_epsilon=convergence_epsilon,\
                             bounds=self.bounds,alpha=[0.0],Lagrangian=True)
        self.graph = self.algo.graph


        #### for second stage (incremental_update)
        self.k_best_solution_set = []
        self.candidate_set = []
        self.current_best_policy = None
        self.candidate_idx = 0   ## this index is used for tie-breaking in heap queue.
        self.candidate_pruning = False
        self.t_start = time.time()
        self.anytime_solutions = []

        self.solution_history = []


    def solve(self, initial_alpha_set):

        return self.find_dual(initial_alpha_set)
 
    def find_dual(self, initial_alpha_set):

        start_time = time.time()

        ##### phase 1
        # zero case
        self.algo.alpha = [initial_alpha_set[0][0]]

        policy = self.algo.solve()
        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))
        self.solution_history.append((weighted_value, (value_1,value_2,value_3), policy))
        
        f_plus = value_1
        g_plus = value_2 - self.bounds[0]

        print("-"*50)
        print("time elapsed: "+str(time.time() - start_time))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_plus))
        print("     g: "+str(g_plus))

        self.add_anytime_solution(f_plus,g_plus)

        if g_plus < 0:
            return True

        while True:
            policy = self.resolve_LAOStar([initial_alpha_set[0][1]])
            value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
            weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

            del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
            self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))
            self.solution_history.append((weighted_value, (value_1,value_2,value_3), policy))

            f_minus = value_1
            g_minus = value_2 - self.bounds[0]

            self.add_anytime_solution(f_minus,g_minus)

            if g_minus>0:
                print("*******************************************")
                print(" bound has been updated ")
                print(g_minus)
                print(initial_alpha_set[0][1])
                print("*******************************************")
                initial_alpha_set[0][1] = initial_alpha_set[0][1]*10

                if initial_alpha_set[0][1] > 10**9:
                    raise ValueError("infeasble")
            else:
                break

        print("-"*50)
        print("time elapsed: "+str(time.time() - start_time))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_minus))
        print("     g: "+str(g_minus))

        # phase 1 interation to compute alpha
        while True:

            # update alpha
            alpha = (f_minus - f_plus) / (g_plus - g_minus)
            L = f_plus + alpha*g_plus
            UB = float('inf')
           
            # evaluate L(u), f, g
            policy = self.resolve_LAOStar([alpha])
            value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
            weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

            del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
            self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))
            self.solution_history.append((weighted_value, (value_1,value_2,value_3), policy))

            L_u = value_1 + alpha*(value_2 - self.bounds[0])
            f = value_1
            g = value_2 - self.bounds[0]

            self.add_anytime_solution(f, g)

            print("-"*50)
            print("time elapsed: "+str(time.time() - start_time))
            print("nodes expanded: "+str(len(self.algo.graph.nodes)))
            print("     f: "+str(f))
            print("     g: "+str(g))
            print("     L: "+str(L_u))

            # cases
            if abs(L_u - L)<0.1**10 and g < 0:
                LB = L_u
                UB = min(f, UB)
                break
            
            elif abs(L_u - L)<0.1**10 and g > 0:
                LB = L_u
                UB = f_minus
                break
            
            elif L_u < L and g > 0:
                f_plus = f
                g_plus = g

            elif L_u < L and g < 0:
                f_minus = f
                g_minus = g
                UB = min(UB, f)

            elif g==0:
                raise ValueError("opt solution found during phase 1.")

            elif L_u > L :
                print(L_u)
                print(L)
                raise ValueError("impossible case. Something must be wrong")



        if LB>=UB:
            ## optimal solutiion found during phase 1
            print("optimal solution found during phase 1!")
 
        else:
            print("dual optima with the following values:")
            print(" alpha:"+str(alpha))
            print("     L: "+str(L))
            print("     f: "+str(f))
            print("     g: "+str(g))
            print("total elapsed time: "+str(time.time()-start_time))
            print("total nodes expanded: "+str(len(self.algo.graph.nodes)))




    def find_dual_multiple_bounds(self, initial_alpha_set):

        k = 0

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 3

        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3

        f_plus = value_1
        g_plus = list()
        g_plus.append(value_2 - self.bounds[0])
        g_plus.append(value_3 - self.bounds[1])

        print("---------- zero case --------")

        # overall infinite case
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3

        f_minus = value_1
        g_minus = list()
        g_minus.append(value_2 - self.bounds[0])
        g_minus.append(value_3 - self.bounds[1])

        print("---------- infinite case --------")

        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]

        L_new = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]
        while True:

            
            L_prev = L_new
            
            for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

                alpha_1_list.append(self.algo.alpha[0])
                alpha_2_list.append(self.algo.alpha[1])
                weighted_value_list.append(f_minus + g_minus[0]*self.algo.alpha[0] + g_minus[1]*self.algo.alpha[1])

                print("*"*20)

                # zero case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
                self.resolve_LAOStar()



                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3

                f_plus = value_1
                g_plus = list()
                g_plus.append(value_2 - self.bounds[0])
                g_plus.append(value_3 - self.bounds[1])


                # infinite case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
                self.resolve_LAOStar()

                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3

                f_minus = value_1
                g_minus = list()
                g_minus.append(value_2 - self.bounds[0])
                g_minus.append(value_3 - self.bounds[1])

                while True:

                    new_alpha_comp = (f_plus - f_minus)

                    for bound_idx_inner in range(len(initial_alpha_set)):
                        if bound_idx_inner==bound_idx:
                            continue

                        new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

                    new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

                    self.algo.alpha[bound_idx] = new_alpha_comp


                    L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1]
                    UB = float('inf')


                    self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3

                    f = value_1
                    g = list()
                    g.append(value_2 - self.bounds[0])
                    g.append(value_3 - self.bounds[1])

                    L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1]

                    

                    print("-"*50)
                    print("time elapsed: "+str(time.time() - self.t_start))
                    print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                    print("     f: "+str(f))
                    print("     g: "+str(g))
                    print("     L: "+str(L_u))


                    # cases
                    if abs(L_u - L)<0.1**10 and g[bound_idx] < 0:
                        LB = L_u
                        UB = min(f, UB)
                        break

                    elif abs(L_u - L)<0.1**10 and g[bound_idx] > 0:
                        LB = L_u
                        UB = f_minus
                        break

                    elif L_u < L and g[bound_idx] > 0:
                        f_plus = f
                        g_plus = g

                    elif L_u < L and g[bound_idx] < 0:
                        f_minus = f
                        g_minus = g
                        UB = min(UB, f)

                    elif g[bound_idx]==0:
                        raise ValueError("opt solution found during phase 1. not implented for this case yet.")

                    elif L_u > L :
                        print(L_u)
                        print(L)
                        raise ValueError("impossible case. Something must be wrong")

                k += 1

            ## optimality check for this entire envelop    

            L_new = L_u

            if abs(L_new - L_prev) < 0.1**5:
                print(alpha_1_list)
                print(alpha_2_list)
                print(weighted_value_list)
                break

            
        print("optimal solution found during phase 1!")
        print("dual optima with the following values:")
        print(" alpha:"+str(self.algo.alpha))
        print("     L: "+str(L))
        print("     f: "+str(f))
        print("     g: "+str(g))

        print(alpha_1_list)
        print(alpha_2_list)
        print(weighted_value_list)





    def find_dual_multiple_bounds_generalized(self, initial_alpha_set, M=6):

        t = time.time()

        k = 0

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 7

        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3
        value_4 = self.algo.graph.root.value_4
        value_5 = self.algo.graph.root.value_5
        value_6 = self.algo.graph.root.value_6
        value_7 = self.algo.graph.root.value_7

        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8, value_9)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        f_plus = value_1
        g_plus = list()
        g_plus.append(value_2 - self.bounds[0])
        g_plus.append(value_3 - self.bounds[1])
        g_plus.append(value_4 - self.bounds[2])
        g_plus.append(value_5 - self.bounds[3])
        g_plus.append(value_6 - self.bounds[4])
        g_plus.append(value_7 - self.bounds[5])

        self.add_anytime_solution(f_plus, g_plus)

        print("---------- zero case --------")
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_plus))
        print("     g: "+str(g_plus))


        # overall infinite case
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3
        value_4 = self.algo.graph.root.value_4
        value_5 = self.algo.graph.root.value_5
        value_6 = self.algo.graph.root.value_6
        value_7 = self.algo.graph.root.value_7

        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8, value_9)

        del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        f_minus = value_1
        g_minus = list()
        g_minus.append(value_2 - self.bounds[0])
        g_minus.append(value_3 - self.bounds[1])
        g_minus.append(value_4 - self.bounds[2])
        g_minus.append(value_5 - self.bounds[3])
        g_minus.append(value_6 - self.bounds[4])
        g_minus.append(value_7 - self.bounds[5])

        self.add_anytime_solution(f_minus, g_minus)

        print("---------- infinite case --------")
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_minus))
        print("     g: "+str(g_minus))

        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]

        L_new = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1] + self.algo.alpha[2]*g_plus[2] \
                       + self.algo.alpha[3]*g_plus[3] + self.algo.alpha[4]*g_plus[4] + self.algo.alpha[5]*g_plus[5]
        while True:

            L_prev = L_new

            for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

                

                # zero case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
                policy = self.resolve_LAOStar()



                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3
                value_4 = self.algo.graph.root.value_4
                value_5 = self.algo.graph.root.value_5
                value_6 = self.algo.graph.root.value_6
                value_7 = self.algo.graph.root.value_7

                f_plus = value_1
                g_plus = list()
                g_plus.append(value_2 - self.bounds[0])
                g_plus.append(value_3 - self.bounds[1])
                g_plus.append(value_4 - self.bounds[2])
                g_plus.append(value_5 - self.bounds[3])
                g_plus.append(value_6 - self.bounds[4])
                g_plus.append(value_7 - self.bounds[5])

                
                self.add_anytime_solution(f_plus, g_plus)

                # infinite case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
                while True:
                    policy = self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3
                    value_4 = self.algo.graph.root.value_4
                    value_5 = self.algo.graph.root.value_5
                    value_6 = self.algo.graph.root.value_6
                    value_7 = self.algo.graph.root.value_7

                    f_minus = value_1
                    g_minus = list()
                    g_minus.append(value_2 - self.bounds[0])
                    g_minus.append(value_3 - self.bounds[1])
                    g_minus.append(value_4 - self.bounds[2])
                    g_minus.append(value_5 - self.bounds[3])
                    g_minus.append(value_6 - self.bounds[4])
                    g_minus.append(value_7 - self.bounds[5])

                    self.add_anytime_solution(f_minus, g_minus)

                    if g_minus[bound_idx]>0:
                        print("*******************************************")
                        print(" bound has been updated ")
                        print(g_minus[bound_idx])
                        print(self.algo.alpha[bound_idx])
                        print("*******************************************")
                        self.algo.alpha[bound_idx] = self.algo.alpha[bound_idx]*10
                    else:
                        break

                while True:

                    if time.time() - t >= 500:
                        return True

                    if g_minus[bound_idx] == g_plus[bound_idx]:
                        f = f_minus
                        g = g_minus
                        L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1] + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3] \
                            + self.algo.alpha[4]*g[4] + self.algo.alpha[5]*g[5]

                        break

                    new_alpha_comp = (f_plus - f_minus)

                    for bound_idx_inner in range(len(initial_alpha_set)):
                        if bound_idx_inner==bound_idx:
                            continue

                        new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

                    new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

                    self.algo.alpha[bound_idx] = new_alpha_comp


                    L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1] + self.algo.alpha[2]*g_plus[2] + self.algo.alpha[3]*g_plus[3] \
                               + self.algo.alpha[4]*g_plus[4] + self.algo.alpha[5]*g_plus[5]
                    UB = float('inf')


                    policy = self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3
                    value_4 = self.algo.graph.root.value_4
                    value_5 = self.algo.graph.root.value_5
                    value_6 = self.algo.graph.root.value_6
                    value_7 = self.algo.graph.root.value_7
                    

                    f = value_1
                    g = list()
                    g.append(value_2 - self.bounds[0])
                    g.append(value_3 - self.bounds[1])
                    g.append(value_4 - self.bounds[2])
                    g.append(value_5 - self.bounds[3])
                    g.append(value_6 - self.bounds[4])
                    g.append(value_7 - self.bounds[5])

                    self.add_anytime_solution(f, g)

                    L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1] + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3] \
                            + self.algo.alpha[4]*g[4] + self.algo.alpha[5]*g[5]

                    # cases
                    if abs(L_u - L)<0.1**10 and g[bound_idx] < 0:
                        LB = L_u
                        UB = min(f, UB)
                        break

                    elif abs(L_u - L)<0.1**10 and g[bound_idx] > 0:
                        LB = L_u
                        UB = f_minus
                        break

                    elif L_u < L and g[bound_idx] > 0:
                        f_plus = f
                        g_plus = g

                    elif L_u < L and g[bound_idx] < 0:
                        f_minus = f
                        g_minus = g
                        UB = min(UB, f)

                    elif g[bound_idx]==0:
                        raise ValueError("opt solution found during phase 1. not implented for this case yet.")

                    elif L_u > L :
                        print(L_u)
                        print(L)
                        raise ValueError("impossible case. Something must be wrong")


                value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
                weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8, value_9)

                del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
                self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

                print("-"*50)
                print("time elapsed: "+str(time.time() - self.t_start))
                print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                print("     f: "+str(f))
                print("     g: "+str(g))
                print("     L: "+str(L_u))


                k += 1


                if k==M:
                    return True

                ## optimality check for this entire envelop    

                L_new = L_u

                if M==np.inf:

                    if abs(L_new - L_prev) < 0.1**5:
                        print("dual optima with the following values:")
                        print(" alpha:"+str(self.algo.alpha))
                        print("     L: "+str(L))
                        print("     f: "+str(f))
                        print("     g: "+str(g))

                        return True
            
                    



 
    def find_dual_multiple_bounds_generalized_const_4(self, initial_alpha_set, M=4):

        k = 0

        alpha_1_list = []
        alpha_2_list = []
        weighted_value_list = []

        self.algo.value_num = 5

        # overall zero case
        self.algo.alpha = [alpha_set[0] for alpha_set in initial_alpha_set]

        policy = self.algo.solve()
        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3
        value_4 = self.algo.graph.root.value_4
        value_5 = self.algo.graph.root.value_5

        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8, value_9)
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        f_plus = value_1
        g_plus = list()
        g_plus.append(value_2 - self.bounds[0])
        g_plus.append(value_3 - self.bounds[1])
        g_plus.append(value_4 - self.bounds[2])
        g_plus.append(value_5 - self.bounds[3])

        self.add_anytime_solution(f_plus, g_plus)

        print("---------- zero case --------")
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_plus))
        print("     g: "+str(g_plus))

        # overall infinite case
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]
        self.resolve_LAOStar()

        value_1 = self.algo.graph.root.value_1
        value_2 = self.algo.graph.root.value_2
        value_3 = self.algo.graph.root.value_3
        value_4 = self.algo.graph.root.value_4
        value_5 = self.algo.graph.root.value_5
        value_6 = self.algo.graph.root.value_6
        value_7 = self.algo.graph.root.value_7

        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8, value_9)

        del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
        self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

        f_minus = value_1
        g_minus = list()
        g_minus.append(value_2 - self.bounds[0])
        g_minus.append(value_3 - self.bounds[1])
        g_minus.append(value_4 - self.bounds[2])
        g_minus.append(value_5 - self.bounds[3])

        self.add_anytime_solution(f_minus, g_minus)

        print("---------- infinite case --------")
        print("time elapsed: "+str(time.time() - self.t_start))
        print("nodes expanded: "+str(len(self.algo.graph.nodes)))
        print("     f: "+str(f_minus))
        print("     g: "+str(g_minus))
        
        self.algo.alpha = [alpha_set[1] for alpha_set in initial_alpha_set]

        L_new = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1] + self.algo.alpha[2]*g_plus[2] \
                       + self.algo.alpha[3]*g_plus[3]
        while True:

            for bound_idx in range(len(initial_alpha_set)):  # looping over each bound (coorindate)

                L_prev = L_new

                # zero case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][0]
                policy = self.resolve_LAOStar()



                value_1 = self.algo.graph.root.value_1
                value_2 = self.algo.graph.root.value_2
                value_3 = self.algo.graph.root.value_3
                value_4 = self.algo.graph.root.value_4
                value_5 = self.algo.graph.root.value_5
                value_6 = self.algo.graph.root.value_6
                value_7 = self.algo.graph.root.value_7

                f_plus = value_1
                g_plus = list()
                g_plus.append(value_2 - self.bounds[0])
                g_plus.append(value_3 - self.bounds[1])
                g_plus.append(value_4 - self.bounds[2])
                g_plus.append(value_5 - self.bounds[3])

                
                self.add_anytime_solution(f_plus, g_plus)

                # infinite case for this coordinate
                self.algo.alpha[bound_idx] = initial_alpha_set[bound_idx][1]
                while True:
                    policy = self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3
                    value_4 = self.algo.graph.root.value_4
                    value_5 = self.algo.graph.root.value_5
                    value_6 = self.algo.graph.root.value_6
                    value_7 = self.algo.graph.root.value_7

                    f_minus = value_1
                    g_minus = list()
                    g_minus.append(value_2 - self.bounds[0])
                    g_minus.append(value_3 - self.bounds[1])
                    g_minus.append(value_4 - self.bounds[2])
                    g_minus.append(value_5 - self.bounds[3])

                    self.add_anytime_solution(f_minus, g_minus)

                    if g_minus[bound_idx]>0:
                        print("*******************************************")
                        print(" bound has been updated ")
                        print(g_minus[bound_idx])
                        print(self.algo.alpha[bound_idx])
                        print("*******************************************")
                        self.algo.alpha[bound_idx] = self.algo.alpha[bound_idx]*10
                    else:
                        break

                while True:

                    if g_minus[bound_idx] == g_plus[bound_idx]:
                        f = f_minus
                        g = g_minus
                        L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1] + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3]

                        break

                    new_alpha_comp = (f_plus - f_minus)

                    for bound_idx_inner in range(len(initial_alpha_set)):
                        if bound_idx_inner==bound_idx:
                            continue

                        new_alpha_comp += self.algo.alpha[bound_idx_inner] * (g_plus[bound_idx_inner] - g_minus[bound_idx_inner])

                    new_alpha_comp = new_alpha_comp / (g_minus[bound_idx]- g_plus[bound_idx])

                    self.algo.alpha[bound_idx] = new_alpha_comp


                    L = f_plus + self.algo.alpha[0]*g_plus[0] + self.algo.alpha[1]*g_plus[1] + self.algo.alpha[2]*g_plus[2] + self.algo.alpha[3]*g_plus[3]
                    UB = float('inf')


                    policy = self.resolve_LAOStar()

                    value_1 = self.algo.graph.root.value_1
                    value_2 = self.algo.graph.root.value_2
                    value_3 = self.algo.graph.root.value_3
                    value_4 = self.algo.graph.root.value_4
                    value_5 = self.algo.graph.root.value_5
                    value_6 = self.algo.graph.root.value_6
                    value_7 = self.algo.graph.root.value_7
                    

                    f = value_1
                    g = list()
                    g.append(value_2 - self.bounds[0])
                    g.append(value_3 - self.bounds[1])
                    g.append(value_4 - self.bounds[2])
                    g.append(value_5 - self.bounds[3])

                    self.add_anytime_solution(f, g)

                    L_u = f + self.algo.alpha[0]*g[0] + self.algo.alpha[1]*g[1] + self.algo.alpha[2]*g[2] + self.algo.alpha[3]*g[3]

                    # cases
                    if abs(L_u - L)<0.1**10 and g[bound_idx] < 0:
                        LB = L_u
                        UB = min(f, UB)
                        break

                    elif abs(L_u - L)<0.1**10 and g[bound_idx] > 0:
                        LB = L_u
                        UB = f_minus
                        break

                    elif L_u < L and g[bound_idx] > 0:
                        f_plus = f
                        g_plus = g

                    elif L_u < L and g[bound_idx] < 0:
                        f_minus = f
                        g_minus = g
                        UB = min(UB, f)

                    elif g[bound_idx]==0:
                        raise ValueError("opt solution found during phase 1. not implented for this case yet.")

                    elif L_u > L :
                        print(L_u)
                        print(L)
                        raise ValueError("impossible case. Something must be wrong")


                value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
                weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8, value_9)

                del self.k_best_solution_set[0]   ## to keep only two solutions at the end. 
                self.k_best_solution_set.append((weighted_value, (value_1,value_2,value_3), policy))

                print("-"*50)
                print("time elapsed: "+str(time.time() - self.t_start))
                print("nodes expanded: "+str(len(self.algo.graph.nodes)))
                print("     f: "+str(f))
                print("     g: "+str(g))
                print("     L: "+str(L_u))


                k += 1


                if k==M:
                    return True

                ## optimality check for this entire envelop    

                L_new = L_u

                if M==np.inf:

                    if abs(L_new - L_prev) < 0.1**5:
                        print("dual optima with the following values:")
                        print(" alpha:"+str(self.algo.alpha))
                        print("     L: "+str(L))
                        print("     f: "+str(f))
                        print("     g: "+str(g))

                        return True
                   




    def resolve_LAOStar(self, new_alpha=None,epsilon=1e-10):

        if new_alpha != None:
            self.algo.alpha = new_alpha
            
        nodes = list(self.algo.graph.nodes.values())
        self.algo.value_iteration(nodes,epsilon=self.resolve_epsilon)
        self.algo.update_fringe()
        return self.algo.solve()


    def incremental_update(self, num_sol, t = time.time(), time_limit=np.inf, cand_method="single", epsilon=0):

        self.algo.incremental = True

        current_best_graph = self.copy_best_graph(self.algo.graph)
        current_best_policy = self.algo.extract_policy()

        self.algo.policy_evaluation(current_best_policy, epsilon=1e-100)

        for k in range(num_sol-1):

            if cand_method == "single":
                candidate_generating_states = self.prune_candidates(current_best_policy, epsilon)

            elif cand_method == "single_wo_speedup":
                candidate_generating_states = self.prune_candidates_wo_speedup(current_best_policy, epsilon)

            elif cand_method == "const6":
                candidate_generating_states = self.prune_candidates_multiple_bounds(current_best_policy, epsilon)

            elif cand_method == "const4":
                candidate_generating_states = self.prune_candidates_multiple_bounds_const4(current_best_policy, epsilon)
            
            

            for state,head,blocked_action_set in candidate_generating_states:

                node = self.algo.graph.nodes[state]
                new_candidate = self.find_candidate(node,head,blocked_action_set)

                if new_candidate:
                    if self.candidate_exists(new_candidate):
                        self.return_to_best_graph(self.algo.graph, current_best_graph)  ## returning to the previous graph.
                        continue
                    else:
                        heappush(self.candidate_set, new_candidate)
                else:
                    continue

                self.return_to_best_graph(self.algo.graph, current_best_graph)  ## returning to the previous graph.

                if time.time() - t >= time_limit:
                    raise ValueError("time limit has met.")

            current_best_graph, current_best_policy = self.find_next_best()
            self.return_to_best_graph(self.algo.graph, current_best_graph)     


    def find_candidate(self, node, head, blocked_action_set):
            
        self.algo.head = head
        self.algo.blocked_action_set = blocked_action_set
        self.algo.tweaking_node = node

        ## if all actions are blocked, then no candidate is generated. 
        if len(self.algo.blocked_action_set)==len(self.model.actions(node.state)):
            return False

        
        self.algo.fringe = set([node])
        policy = self.algo.solve()

        value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9 = self.algo.get_values(self.algo.graph.root)
        weighted_value = self.algo.compute_weighted_value(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9)

        self.add_anytime_solution(value_1, value_2 - self.bounds[0])

        self.candidate_idx += 1

        new_best_graph = self.copy_best_graph(self.algo.graph)

        return (weighted_value, self.candidate_idx, (value_1,value_2,value_3), new_best_graph, policy)
        # return (weighted_value, self.candidate_idx, (value_1,value_2,value_3), self.copy_graph(self.algo.graph), policy)



    def get_blocked_action_set(self,node,head):

        blocked_action_set = set()

        ## TODO: current policy need not be inspected in this way. Can be more easily done. Fix it for better efficiency later.
        for solution in self.k_best_solution_set:
            
            prev_policy = solution[2]

            if self.is_head_eq(head, prev_policy):
                blocked_action_set.add(prev_policy[node.state])

        return blocked_action_set


    def get_head(self, node):

        queue = set([self.algo.graph.root])
        head = set()

        while queue:

            popped_node = queue.pop()

            if popped_node in head:
                continue

            if popped_node == node:
                continue

            if popped_node.terminal==True:
                continue

            head.add(popped_node)
            
            if popped_node.best_action!=None:
                children = popped_node.children[popped_node.best_action]

                for child,child_prob in children:
                    queue.add(child)

            else:
                raise ValueError("Policy seems have unexpanded node.")                    

        return head    
    

    
    def get_ancestors(self, node):
        Z = set()

        queue = set([node])

        while queue:
            node = queue.pop()

            if node not in Z:
                
                Z.add(node)
                parents = node.best_parents_set

                queue = queue.union(parents)

        return Z

    

    def is_head_eq(self,head,policy):

        for node in head:

            if node.terminal==True:
                continue
            
            if node.state not in policy:
                return False

            else:
                if node.best_action == policy[node.state]:
                    continue
                else:
                    return False

        return True
    

    def candidate_exists(self,new_candidate):

        for candidate in self.candidate_set:

            # if abs(candidate[0] - new_candidate[0]) > 1e-5:
            #     continue

            # else:
            if self.is_policy_eq(candidate[4], new_candidate[4]):
                return True


        return False


    def is_policy_eq(self, policy_1, policy_2):

        if len(policy_1)!=len(policy_2):
            return False

        else:

            for state, action in policy_1.items():

                if state not in policy_2:
                    return False
                else:
                    if action != policy_2[state]:
                        return False

        return True

        
    def find_next_best(self):
        
        next_best_candidate = heappop(self.candidate_set)

        current_best_weighted_value = next_best_candidate[0]
        current_best_values = next_best_candidate[2]
        current_best_graph = next_best_candidate[3]
        current_best_policy = next_best_candidate[4]

        self.k_best_solution_set.append((current_best_weighted_value, current_best_values, current_best_policy))

        return current_best_graph, current_best_policy




    def copy_graph(self,graph):

        new_graph = Graph(name='G')

        for state, node in graph.nodes.items():

            new_graph.add_node(state, node.value_1, node.value_2, node.value_3, node.best_action, node.terminal)

        for state, node in graph.nodes.items():
            for action, children in node.children.items():
                new_graph.nodes[state].children[action] = []
                for child, child_prob in children:
                    new_graph.nodes[state].children[action].append([new_graph.nodes[child.state], child_prob])

            for parents_node in node.best_parents_set:
                new_graph.nodes[state].best_parents_set.add(new_graph.nodes[parents_node.state])
                
        new_graph.root = new_graph.nodes[self.model.init_state]

        return new_graph




    def copy_best_graph(self, graph):

        copied_best_graph = dict()

        queue = set([self.graph.root])

        while queue:

            node = queue.pop()

            if node.state in copied_best_graph:
                continue

            else:
                if node.best_action!=None:
                    copied_best_graph[node.state] = (node.best_action, node.value_1, node.value_2, node.value_3, node.value_4, \
                                                     node.value_5, node.value_6, node.value_7, node.children, node.best_parents_set)
                    children = node.children[node.best_action]

                    for child,child_prob in children:
                        queue.add(child)

                elif node.terminal==True:
                    copied_best_graph[node.state] = (node.best_action, node.value_1, node.value_2, node.value_3, node.value_4, \
                                                     node.value_5, node.value_6, node.value_7, node.children, node.best_parents_set)
                else:
                    raise ValueError("Best partial graph has non-expanded fringe node.")

        return copied_best_graph


    
    def return_to_best_graph(self, graph, copied_best_graph):

        for state, contents in copied_best_graph.items():

            node = graph.nodes[state]
            
            node.best_action = contents[0]
            node.value_1 = contents[1]
            node.value_2 = contents[2]
            node.value_3 = contents[3]
            node.value_4 = contents[4]
            node.value_5 = contents[5]
            node.value_6 = contents[6]
            node.value_7 = contents[7]
            node.children = contents[8]
            node.best_parents_set = contents[9]






    def prune_candidates(self, current_best_policy, given_epsilon):

        if self.candidate_pruning==False:
            candidate_generating_states = []
            
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        
        elif (self.algo.graph.root.value_2 - self.bounds[0]) <= 0:
            epsilon = given_epsilon
            initial_epsilon = 1e-4

            policy_value = self.algo.graph.root.value_1

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]
            R = np.ones(num_states)  

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]
            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value) / policy_value

            prev_value = new_value

            if epsilon < 0:
                print("initial pruning was too aggressive!")
            else:
                N = N_new

            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R[idx] = 0
                    new_value = np.dot(N_new[root_idx,:], R)


                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            continue
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) / policy_value < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value) / policy_value
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = 1
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states

        

        elif (self.algo.graph.root.value_2 - self.bounds[0]) > 0:
            policy_value = self.algo.graph.root.value_2
            epsilon = policy_value - self.bounds[0]
            initial_epsilon = 1e-3
            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)
            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]

            R = self.compute_R(state_list, current_best_policy)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]


            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value)

            prev_value = new_value

            if epsilon < 0:
                print("initial pruning was too aggressive!")
            else:
                N = N_new

            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R_prev = R[idx]
                    R[idx] = 0
                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            continue
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value)
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = R_prev
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states






    def prune_candidates_wo_speedup(self, current_best_policy):

        if self.candidate_pruning==False:
            candidate_generating_states = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        
        elif (self.algo.graph.root.value_2 - self.bounds[0]) <= 0:
            epsilon = 0.1
            initial_epsilon = 1e-4

            policy_value = self.algo.graph.root.value_1

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]
            R = np.ones(num_states) 

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)

            epsilon -= (policy_value - new_value) / policy_value

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    Q[idx,:] = np.zeros(num_states)
                    R[idx] = 0
                    N_new = np.linalg.inv(I - Q)
                    V = np.dot(N_new, R)
                    new_value = V[root_idx]

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) / policy_value < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value) / policy_value
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = 1
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states

        

        elif (self.algo.graph.root.value_2 - self.bounds[0]) > 0:

            policy_value = self.algo.graph.root.value_2
            
            epsilon = policy_value - self.bounds[0]
            initial_epsilon = 1e-3
            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)

            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            N = np.linalg.inv(I - Q)
            N_vector = N[root_idx]

            R = self.compute_R(state_list, current_best_policy)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]


            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value)

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    Q[idx,:] = np.zeros(num_states)
                    R_prev = R[idx]
                    R[idx] = 0
                    N_new = np.linalg.inv(I - Q)
                    V = np.dot(N_new, R)
                    new_value = V[root_idx]

                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value)
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = R_prev
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states
        




    def prune_candidates_multiple_bounds(self, current_best_policy, given_epsilon):

        Feasible = True

        g = []
        g.append(self.algo.graph.root.value_2 - self.bounds[0])
        g.append(self.algo.graph.root.value_3 - self.bounds[1])
        g.append(self.algo.graph.root.value_4 - self.bounds[2])
        g.append(self.algo.graph.root.value_5 - self.bounds[3])
        g.append(self.algo.graph.root.value_6 - self.bounds[4])
        g.append(self.algo.graph.root.value_7 - self.bounds[5])

        for g_elem in g:
            if g_elem>0:
                Feasible = False
                break

        if self.candidate_pruning==False:
            candidate_generating_states = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        elif Feasible == True:
            epsilon = given_epsilon

            initial_epsilon = 1e-1
            policy_value = self.algo.graph.root.value_1

            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)

            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)

            try:
                N = np.linalg.inv(I - Q)
            except:
                candidate_generating_states = []
                for state, action in current_best_policy.items():
                    if action != 'Terminal':

                        node = self.algo.graph.nodes[state]
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                    
                        candidate_generating_states.append((state,head,blocked_action_set))

                return candidate_generating_states
                
            N_vector = N[root_idx]
            R = np.ones(num_states)  

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]
            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)

            epsilon -= (policy_value - new_value) / policy_value

            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new

            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R[idx] = 0
                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) / policy_value < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value) / policy_value
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = 1
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states

        

        else:

            g = []
            g.append(self.algo.graph.root.value_2 - self.bounds[0])
            g.append(self.algo.graph.root.value_3 - self.bounds[1])
            g.append(self.algo.graph.root.value_4 - self.bounds[2])
            g.append(self.algo.graph.root.value_5 - self.bounds[3])
            g.append(self.algo.graph.root.value_6 - self.bounds[4])
            g.append(self.algo.graph.root.value_7 - self.bounds[5])

            inf_ind = 0
            k=0
            max_g_elem = -1
            for g_elem in g:
                if g_elem>0:
                    if g_elem > max_g_elem:
                        max_g_elem = g_elem
                        inf_ind = k

            policy_value = g[inf_ind] + self.bounds[inf_ind]
            
            epsilon = policy_value - self.bounds[inf_ind]
            initial_epsilon = 1e-4


            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            try:
                N = np.linalg.inv(I - Q)
            except:
                candidate_generating_states = []
                for state, action in current_best_policy.items():
                    if action != 'Terminal':

                        node = self.algo.graph.nodes[state]
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                    
                        candidate_generating_states.append((state,head,blocked_action_set))

                return candidate_generating_states

            
            N_vector = N[root_idx]

            R = self.compute_R_multiple_bounds(state_list,inf_ind=inf_ind)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]


            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)


            epsilon -= (policy_value - new_value)

            prev_value = new_value

            if epsilon < 0:

                print("initial pruning was too aggressive 1")
                candidate_generating_states = []
                for state, action in current_best_policy.items():
                    if action != 'Terminal':

                        node = self.algo.graph.nodes[state]
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                    
                        candidate_generating_states.append((state,head,blocked_action_set))

                return candidate_generating_states

            else:
                N = N_new


            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R_prev = R[idx]
                    R[idx] = 0
                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value)
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = R_prev
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states




    def prune_candidates_multiple_bounds_const4(self, current_best_policy, given_epsilon):

        Feasible = True

        g = []
        g.append(self.algo.graph.root.value_2 - self.bounds[0])
        g.append(self.algo.graph.root.value_3 - self.bounds[1])
        g.append(self.algo.graph.root.value_4 - self.bounds[2])
        g.append(self.algo.graph.root.value_5 - self.bounds[3])

        for g_elem in g:
            if g_elem>0:
                Feasible = False
                break

        if self.candidate_pruning==False:
            candidate_generating_states = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':

                    node = self.algo.graph.nodes[state]
                    head = self.get_head(node)
                    blocked_action_set = self.get_blocked_action_set(node,head)
                    
                    candidate_generating_states.append((state,head,blocked_action_set))

            return candidate_generating_states
        
        elif Feasible == True:
            epsilon = given_epsilon
            initial_epsilon = 1e-1
            policy_value = self.algo.graph.root.value_1
            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)

            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)

            try:
                N = np.linalg.inv(I - Q)
            except:
                candidate_generating_states = []
                for state, action in current_best_policy.items():
                    if action != 'Terminal':

                        node = self.algo.graph.nodes[state]
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                    
                        candidate_generating_states.append((state,head,blocked_action_set))

                return candidate_generating_states
                
            N_vector = N[root_idx]
            R = np.ones(num_states)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]
            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)
            epsilon -= (policy_value - new_value) / policy_value
            prev_value = new_value

            if epsilon < 0:
                raise ValueError("initial pruning was too aggressive!")
            else:
                N = N_new

            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R[idx] = 0
                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) / policy_value < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value) / policy_value
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = 1
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states

        else:

            g = []
            g.append(self.algo.graph.root.value_2 - self.bounds[0])
            g.append(self.algo.graph.root.value_3 - self.bounds[1])
            g.append(self.algo.graph.root.value_4 - self.bounds[2])
            g.append(self.algo.graph.root.value_5 - self.bounds[3])

            inf_ind = 0
            k=0
            max_g_elem = -1
            for g_elem in g:
                if g_elem>0:
                    if g_elem > max_g_elem:
                        max_g_elem = g_elem
                        inf_ind = k

            policy_value = g[inf_ind] + self.bounds[inf_ind]
            
            epsilon = policy_value - self.bounds[inf_ind]
            initial_epsilon = 1e-4


            state_list = []
            for state, action in current_best_policy.items():
                if action != 'Terminal':
                    state_list.append(state)


            num_states = len(state_list)

            Q,root_idx,state_idx_dict = self.compute_transition_matrix(state_list)
            I = np.identity(num_states)
            try:
                N = np.linalg.inv(I - Q)
            except:
                candidate_generating_states = []
                for state, action in current_best_policy.items():
                    if action != 'Terminal':

                        node = self.algo.graph.nodes[state]
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                    
                        candidate_generating_states.append((state,head,blocked_action_set))

                return candidate_generating_states

            
            N_vector = N[root_idx]

            R = self.compute_R_multiple_bounds(state_list,inf_ind=inf_ind)

            sorted_states = [state for _, state in sorted(zip(N_vector, state_list))]

            initial_pruned_states = [(state,state_idx_dict[state]) for state in sorted_states if N_vector[state_idx_dict[state]]<initial_epsilon]

            for pruned_state in initial_pruned_states:
                state = pruned_state[0]
                idx = pruned_state[1]
                Q[idx,:] = np.zeros(num_states)
                R[idx] = 0

            N_new = np.linalg.inv(I - Q)
            new_value = np.dot(N_new[root_idx,:], R)

            epsilon -= (policy_value - new_value)

            prev_value = new_value

            if epsilon < 0:

                print("initial pruning was too aggressive 1")
                candidate_generating_states = []
                for state, action in current_best_policy.items():
                    if action != 'Terminal':

                        node = self.algo.graph.nodes[state]
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                    
                        candidate_generating_states.append((state,head,blocked_action_set))

                return candidate_generating_states
            else:
                N = N_new

            candidate_generating_states = []
            accumulated_head = set(self.algo.graph.nodes.values())
            pruned_states = []

            for state in sorted_states:
                idx = state_idx_dict[state]

                if (state,idx) in initial_pruned_states:
                    continue

                node = self.algo.graph.nodes[state]

                if node not in accumulated_head:
                    continue

                else:
                    u = np.zeros(num_states)
                    u[idx] = 1
                    v = Q[idx,:]
                    N_new = self.SM_update(N,u,v)
                    R_prev = R[idx]
                    R[idx] = 0
                    new_value = np.dot(N_new[root_idx,:], R)

                    if prev_value < new_value:
                        if abs(prev_value - new_value) > 1e-8:
                            print(prev_value)
                            print(new_value)
                            raise ValueError("something went wrong.")

                    elif (prev_value - new_value) < epsilon:
                        ## can be pruned
                        N = N_new
                        epsilon -= (prev_value - new_value)
                        head = self.get_head(node)
                        accumulated_head = accumulated_head.intersection(head)
                        prev_value = new_value
                        pruned_states.append(state)

                    else:
                        ## cannot be pruned
                        R[idx] = R_prev
                        head = self.get_head(node)
                        blocked_action_set = self.get_blocked_action_set(node,head)
                        candidate_generating_states.append((state, head, blocked_action_set))

            return candidate_generating_states        



    def compute_transition_matrix(self, state_list):

        num_states = len(state_list)

        Q = np.empty((num_states, num_states))
        root_idx = None

        i = 0
        state_idx_dict = dict()
        for state in state_list:
            state_idx_dict[state] = i

            if state==self.model.init_state:
                root_idx = state_list.index(state)

            Q_vector = np.zeros(num_states)

            node = self.algo.graph.nodes[state]
            children = node.children[node.best_action]

            for child, child_prob in children:
                if child.terminal != True:
                    idx = state_list.index(child.state)
                    if Q_vector[idx] > 0:
                        Q_vector[idx] += child_prob
                    else:
                        Q_vector[idx] = child_prob

            Q[i,:] = Q_vector

            i += 1

        return Q, root_idx, state_idx_dict
   



    def SM_update(self,B, u, v):
        return B - np.outer(B @ u, v @ B) / (1 + v.T @ B @ u)



    def compute_R(self,state_list,current_best_policy=None, inf_ind=None):
        
        R = np.zeros(len(state_list))

        i = 0
        for state in state_list:
            if current_best_policy!=None:
                cost1, cost2 = self.model.cost(state, current_best_policy[state])
            else:
                cost1, cost2 = self.model.cost(state, None)
            R[i] = cost2
            i += 1


        return R

    def compute_R_multiple_bounds(self,state_list,current_best_policy=None, inf_ind=None):
        
        R = np.zeros(len(state_list))

        i = 0


        for state in state_list:
            cost1, cost2, cost3, cost4, cost5, cost6, cost7 = self.model.cost(state, None)
            cost_vec = [cost2, cost3, cost4, cost5, cost6, cost7]
            R[i] = cost_vec[inf_ind]
            i += 1

        return R
            

    def compute_likelihoods(self, state_list):

        num_states = len(state_list)
        
        Q = np.empty((num_states, num_states))
        root_idx = None

        i = 0
        for state in state_list:

            if state==self.model.init_state:
                root_idx = state_list.index(state)

            Q_vector = np.zeros(num_states)

            node = self.algo.graph.nodes[state]
            children = node.children[node.best_action]

            for child, child_prob in children:
                if child.terminal != True:
                    idx = state_list.index(child.state)
                    Q_vector[idx] = child_prob

            Q[i,:] = Q_vector
            
            i += 1


        t = time.time()
        I = np.identity(num_states)
        o = np.zeros(num_states)
        o[root_idx] = 1
        L = np.linalg.solve(np.transpose(I-Q), o)
        L = L / L[root_idx]

        return L  
    

    def add_anytime_solution(self, f, g):

        if type(g)==list:
            feasible=True
            for g_val in g:
                if g_val>0:
                    feasible=False
                    break
                
            if feasible==True:
                if len(self.anytime_solutions)==0:
                    self.anytime_solutions.append((f, time.time() - self.t_start, len(self.algo.graph.nodes)))
                    print("@@@@@@@@@@@@@@@@@@ New anytime solution found:")
                    print(self.anytime_solutions)
                    print(len(self.algo.graph.nodes))

                else:

                    if self.anytime_solutions[-1][0] > f:
                        self.anytime_solutions.append((f, time.time() - self.t_start, len(self.algo.graph.nodes)))
                        print("@@@@@@@@@@@@@@@@@@@@@@@@ New anytime solution found:")
                        print(self.anytime_solutions)
                        print(len(self.algo.graph.nodes))

        else:
            if g<0:
                if len(self.anytime_solutions)==0:
                    self.anytime_solutions.append((f, time.time() - self.t_start, len(self.algo.graph.nodes)))
                    print("@@@@@@@@@@@@@@@@@@@@@@@@ New anytime solution found:")
                    print(self.anytime_solutions)
                    print(len(self.algo.graph.nodes))

                else:

                    if self.anytime_solutions[-1][0] > f:
                        self.anytime_solutions.append((f, time.time() - self.t_start, len(self.algo.graph.nodes)))
                        print("@@@@@@@@@@@@@@@@@@@@@@@@ New anytime solution found:")
                        print(self.anytime_solutions)
                        print(len(self.algo.graph.nodes))




    def is_feasible(self, f, g):

        if type(g)==list:
            feasible=True
            for g_val in g:
                if g_val>0:
                    feasible=False
                    break
                
            if feasible==True:
                return True
            else:
                return False

        else:
            if g<=0:
                return True

            else:
                return False

