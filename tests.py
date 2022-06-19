#!/usr/bin/env python

import sys
from utils import import_models
import_models()

from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar
from value_iteration import VI
from CSSPSolver import CSSPSolver
from racetrack_model import RaceTrackModel
from elevator_const_6 import ELEVATORModel_1
from elevator_const_4 import ELEVATORModel_2
from cc_routing_model import CCROUTINGModel

import numpy as np
import time
import random
import cProfile
import json

def linspace(start, stop, n):
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i
        

def racetrack_large():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_large_a.txt"
    traj_check_dict_file = "models/racetrack_large_traj_check_dict.json"
    heuristic_file = "models/racetrack_large_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)

    bound = 1

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    t = time.time()

    cssp_solver.solve([[0,1.0]])

    policy = cssp_solver.algo.extract_policy()

    cssp_solver.candidate_pruning = False

    try:
        cssp_solver.incremental_update(2, cand_method="single", epsilon=0)
    except:

        k_best_solution_set = cssp_solver.k_best_solution_set
        for solution in k_best_solution_set:
            print("-"*20)
            print(solution[0])
            print(solution[1])

        print(time.time() - t)

        print(cssp_solver.anytime_solutions)


    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])

    print(time.time() - t)

    print(cssp_solver.anytime_solutions)



def racetrack_ring():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_ring_a.txt"
    traj_check_dict_file = "models/racetrack_ring_traj_check_dict.json"
    heuristic_file = "models/racetrack_ring_heuristic.json"

    init_state = (1,23,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)

    bound = 1

    cssp_solver = CSSPSolver(model, bounds=[bound],VI_epsilon=1e-1,convergence_epsilon=1e-10)

    t = time.time()

    cssp_solver.solve([[0,1.0]])

    policy = cssp_solver.algo.extract_policy()

    cssp_solver.candidate_pruning = False

    try:
        cssp_solver.incremental_update(2, cand_method="single", epsilon=0)
    except:

        k_best_solution_set = cssp_solver.k_best_solution_set
        for solution in k_best_solution_set:
            print("-"*20)
            print(solution[0])
            print(solution[1])

        print(time.time() - t)

        print(cssp_solver.anytime_solutions)


    k_best_solution_set = cssp_solver.k_best_solution_set
    for solution in k_best_solution_set:
        print("-"*20)
        print(solution[0])
        print(solution[1])

    print(time.time() - t)

    print(cssp_solver.anytime_solutions)
            




def elevators_const_6():

    init_state = ((7, 19), (0,), (6, 0), (16, 0))
    px_dest = (1, 4)
    hidden_dest = (6,)
    hidden_origin = (8,)

    model = ELEVATORModel_1(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                                 px_dest=px_dest, \
                                 hidden_dest=hidden_dest, \
                                 hidden_origin=hidden_origin)

    alpha = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [15,21,15,21,15,21]

    M = 6
    epsilon = 0.3

    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-100)

    t = time.time()

    try:
        cssp_solver.find_dual_multiple_bounds_generalized([[0,.1],[0,.1],[0,.1],[0,.1],[0,.1],[0,.1]], M=M)
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)

    cssp_solver.candidate_pruning = True

    try:
        cssp_solver.incremental_update(1000, t, 500, cand_method="const6", epsilon=epsilon)
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)


    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))

    print(cssp_solver.anytime_solutions)




def elevators_const_4():

    init_state = ((9, 11), (0,), (3, 0), (19, 0))
    px_dest = (18, 16)
    hidden_dest = (9,)
    hidden_origin = (8,)

    model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                            px_dest=px_dest, \
                            hidden_dest=hidden_dest, \
                            hidden_origin=hidden_origin)

    alpha = [0.0, 0.0, 0.0, 0.0]
    bounds = [15,15,15,21]

    M = 4
    epsilon = 0.3

    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-1,convergence_epsilon=1e-100)

    t = time.time()

    try:
        cssp_solver.find_dual_multiple_bounds_generalized_const_4([[0,.1],[0,.1],[0,.1],[0,.1]], M=M)
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)

    cssp_solver.candidate_pruning = True

    try:
        cssp_solver.incremental_update(1000, t, 500, cand_method="const4", epsilon=epsilon)
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))

    print(cssp_solver.anytime_solutions)

    


def cc_routing_config_1():

    obs_init_state = (8, 8)
    obs_dir = ['L']

    init_state = ((0,0),obs_init_state)
    size = (15,10)
    goal = (5,5)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=obs_dir, obs_boundary=[(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]   

    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-100,convergence_epsilon=1e-300)

    t = time.time()

    solved = True

    try:
        solved = cssp_solver.solve([[0,0.1]])
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)

    cssp_solver.candidate_pruning = True

    if solved != True:
        try:
            cssp_solver.incremental_update(300, t, 300, cand_method="single", epsilon=0.1)
            # cssp_solver.incremental_update(300, t, 300, cand_method="single_wo_speedup", epsilon=0.1)
        except:
            print(time.time() - t)
            print(cssp_solver.anytime_solutions)

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))

    print(cssp_solver.anytime_solutions)



def cc_routing_config_2():

    obs_init_state = (7, 13)
    obs_dir = ['L']

    init_state = ((0,0),obs_init_state)
    size = (15,15)
    goal = (13,13)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=obs_dir, obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]   

    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-100,convergence_epsilon=1e-300)

    t = time.time()

    solved = True

    try:
        solved = cssp_solver.solve([[0,0.1]])
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)

    cssp_solver.candidate_pruning = True

    if solved != True:
        try:
            cssp_solver.incremental_update(300, t, 1800, cand_method="single", epsilon=0.1)
            # cssp_solver.incremental_update(300, t, 300, cand_method="single_wo_speedup", epsilon=0.1)
        except:
            print(time.time() - t)
            print(cssp_solver.anytime_solutions)

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))

    print(cssp_solver.anytime_solutions)


def cc_routing_config_3():

    obs_init_state = [(7, 6), (14, 8)]
    obs_dir = ['L', 'R']

    init_state = ((0,0),obs_init_state[0],obs_init_state[1])
    size = (15,10)
    goal = (13,8)
    model = CCROUTINGModel(size, obs_num=2, obs_dir=obs_dir, obs_boundary=[(2,1),(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]   

    cssp_solver = CSSPSolver(model, bounds=bounds,VI_epsilon=1e-100,convergence_epsilon=1e-300)

    t = time.time()

    solved = True

    try:
        solved = cssp_solver.solve([[0,0.1]])
    except:
        print(time.time() - t)
        print(cssp_solver.anytime_solutions)


    cssp_solver.candidate_pruning = False

    if solved != True:
        try:
            cssp_solver.incremental_update(300, t, 1800, cand_method="single", epsilon=0.1)
            # cssp_solver.incremental_update(300, t, 300, cand_method="single_wo_speedup", epsilon=0.1)
        except:
            print(time.time() - t)
            print(cssp_solver.anytime_solutions)

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(cssp_solver.graph.nodes)))

    print(cssp_solver.anytime_solutions)



racetrack_large()
# racetrack_ring()

# elevators_const_6()
# elevators_const_4()

# cc_routing_config_1()
# cc_routing_config_2()
# cc_routing_config_3()
