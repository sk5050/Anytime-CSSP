#!/usr/bin/env python

import sys
from utils import import_models
import_models()

from graph import Node, Graph
from LAOStar import LAOStar
from ILAOStar import ILAOStar
from value_iteration import VI
from CSSPSolver import CSSPSolver
from MILPSolver import MILPSolver
from IDUAL import IDUAL


from racetrack_model import RaceTrackModel
from elevator_const_6 import ELEVATORModel_1
from elevator_const_4 import ELEVATORModel_2
from cc_routing_model import CCROUTINGModel

import numpy as np
import time
import random
import cProfile
import json


def racetrack_large_MILP():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_large_a.txt"
    traj_check_dict_file = "models/racetrack_large_traj_check_dict.json"
    heuristic_file = "models/racetrack_large_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    bound = [1]

    solver = MILPSolver(model, bound)

    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))


def racetrack_ring_MILP():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_ring_a.txt"
    traj_check_dict_file = "models/racetrack_ring_traj_check_dict.json"
    heuristic_file = "models/racetrack_ring_heuristic.json"

    init_state = (1,23,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)
    bound = [1]

    solver = MILPSolver(model, bound)

    t = time.time()
    solver.solve_opt_MILP()
    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))


def racetrack_large_IDUAL():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_large_a.txt"
    traj_check_dict_file = "models/racetrack_large_traj_check_dict.json"
    heuristic_file = "models/racetrack_large_heuristic.json"

    init_state = (3,1,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)

    bounds = [1]

    t = time.time()
    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))


def racetrack_ring_IDUAL():

    sys.setrecursionlimit(8000)

    map_file = "models/racetrack_ring_a.txt"
    traj_check_dict_file = "models/racetrack_ring_traj_check_dict.json"
    heuristic_file = "models/racetrack_ring_heuristic.json"

    init_state = (1,23,0,0)
    model = RaceTrackModel(map_file, init_state=init_state, traj_check_dict_file=traj_check_dict_file, heuristic_file=heuristic_file, slip_prob=0.1)

    bounds = [1]

    t = time.time()
    solver = IDUAL(model, bounds)
    solver.solve_LP_and_MILP()

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))


def elevator_const6_MILP():

    init_state = ((7, 19), (0,), (6, 0), (16, 0))
    px_dest = (1, 4)
    hidden_dest = (6,)
    hidden_origin = (8,)

    model = ELEVATORModel_1(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                            px_dest=px_dest, \
                            hidden_dest=hidden_dest, \
                            hidden_origin=hidden_origin)

    bounds = [15,21,15,21,15,21]

    solver = MILPSolver(model, bounds)

    t = time.time()

    try:
        objVal = solver.solve_opt_MILP()
    except:
        objVal = None

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))


def elevator_const4_MILP():

    init_state = ((9, 11), (0,), (3, 0), (19, 0))
    px_dest = (18, 16)
    hidden_dest = (9,)
    hidden_origin = (8,)

    model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                            px_dest=px_dest, \
                            hidden_dest=hidden_dest, \
                            hidden_origin=hidden_origin)

    bounds = [15,15,15,21]

    solver = MILPSolver(model, bounds)

    t = time.time()

    try:
        objVal = solver.solve_opt_MILP()
    except:
        objVal = None

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))


def elevator_const6_IDUAL():

    init_state = ((7, 19), (0,), (6, 0), (16, 0))
    px_dest = (1, 4)
    hidden_dest = (6,)
    hidden_origin = (8,)

    model = ELEVATORModel_1(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                            px_dest=px_dest, \
                            hidden_dest=hidden_dest, \
                            hidden_origin=hidden_origin)

    bounds = [15,21,15,21,15,21]

    t = time.time()

    solver = IDUAL(model, bounds)

    try:
        objVal, lp_time, lp_nodes = solver.solve_LP_and_MILP(t,time_limit=1800)
    except:
        objVal = np.inf
        lp_time = solver.lp_time
        lp_nodes = solver.lp_nodes

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))


def elevator_const4_IDUAL():

    init_state = ((9, 11), (0,), (3, 0), (19, 0))
    px_dest = (18, 16)
    hidden_dest = (9,)
    hidden_origin = (8,)

    model = ELEVATORModel_2(n=20, w=2, h=1, prob=0.75, init_state=init_state, \
                            px_dest=px_dest, \
                            hidden_dest=hidden_dest, \
                            hidden_origin=hidden_origin)

    bounds = [15,15,15,21]

    t = time.time()

    solver = IDUAL(model, bounds)

    try:
        objVal, lp_time, lp_nodes = solver.solve_LP_and_MILP(t,time_limit=1800)
    except:
        objVal = np.inf
        lp_time = solver.lp_time
        lp_nodes = solver.lp_nodes

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))


def cc_routing_config_1_MILP():

    obs_init_state = (8, 8)
    obs_dir = ['L']

    init_state = ((0,0),obs_init_state)
    size = (15,10)
    goal = (5,5)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=obs_dir, obs_boundary=[(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]

    solver = MILPSolver(model, bounds)

    t = time.time()

    try:
        objVal = solver.solve_opt_MILP()
    except:
        objVal = None

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))



def cc_routing_config_2_MILP():

    obs_init_state = (7, 13)
    obs_dir = ['L']

    init_state = ((0,0),obs_init_state)
    size = (15,15)
    goal = (13,13)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=obs_dir, obs_boundary=[(2,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]

    solver = MILPSolver(model, bounds)

    t = time.time()

    try:
        objVal = solver.solve_opt_MILP(time_limit=1800)
    except:
        objVal = None

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))



def cc_routing_config_3_MILP():

    obs_init_state = [(7, 6), (14, 8)]
    obs_dir = ['L', 'R']

    init_state = ((0,0),obs_init_state[0],obs_init_state[1])
    size = (15,10)
    goal = (13,8)
    model = CCROUTINGModel(size, obs_num=2, obs_dir=obs_dir, obs_boundary=[(2,1),(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]

    solver = MILPSolver(model, bounds)

    t = time.time()

    try:
        objVal = solver.solve_opt_MILP(time_limit=1800)
    except:
        objVal = None

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.algo.graph.nodes)))



def cc_routing_config_1_IDUAL():

    obs_init_state = (8, 8)
    obs_dir = ['L']

    init_state = ((0,0),obs_init_state)
    size = (15,10)
    goal = (5,5)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=obs_dir, obs_boundary=[(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]

    t = time.time()
    solver = IDUAL(model, bounds)

    objVal, lp_time, lp_nodes = solver.solve_LP_and_MILP(t, time_limit=1800)

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))




def cc_routing_config_2_IDUAL():

    obs_init_state = (7, 13)
    obs_dir = ['L']

    init_state = ((0,0),obs_init_state)
    size = (15,15)
    goal = (13,13)
    model = CCROUTINGModel(size, obs_num=1, obs_dir=obs_dir, obs_boundary=[(1,1)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]

    t = time.time()
    solver = IDUAL(model, bounds)

    try:
        objVal, lp_time, lp_nodes = solver.solve_LP_and_MILP(t, time_limit=1800)
    except:
        objVal = np.inf
        lp_time = solver.lp_time
        lp_nodes = solver.lp_nodes

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))



def cc_routing_config_3_IDUAL():

    obs_init_state = [(7, 6), (14, 8)]
    obs_dir = ['L', 'R']

    init_state = ((0,0),obs_init_state[0],obs_init_state[1])
    size = (15,10)
    goal = (13,8)
    model = CCROUTINGModel(size, obs_num=2, obs_dir=obs_dir, obs_boundary=[(2,1),(1,2)], init_state=init_state, goal=goal, prob_right_transition=0.8)

    bounds = [0.01]

    t = time.time()
    solver = IDUAL(model, bounds)

    try:
        objVal, lp_time, lp_nodes = solver.solve_LP_and_MILP(t, time_limit=1800)
    except:
        objVal = np.inf
        lp_time = solver.lp_time
        lp_nodes = solver.lp_nodes

    print("elapsed time: "+str(time.time()-t))
    print("number of states explored: "+str(len(solver.graph.nodes)))



#################################################
################### MILP Tests ##################
#################################################

racetrack_large_MILP()
# racetrack_ring_MILP()

# elevator_const6_MILP()
# elevator_const4_MILP()

# cc_routing_config_1_MILP()
# cc_routing_config_2_MILP()
# cc_routing_config_3_MILP()


#################################################
################### i-dual Tests ################
#################################################

# racetrack_large_IDUAL()
# racetrack_ring_IDUAL()

# elevator_const6_IDUAL()
# elevator_const4_IDUAL()

# cc_routing_config_1_IDUAL()
# cc_routing_config_2_IDUAL()
# cc_routing_config_3_IDUAL()
