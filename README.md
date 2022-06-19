# An Anytime Algorithm For C-SSP With Deterministic Policies

This repository contains the implementation of anytime algorithm for C-SSP. The paper is under review and will be updated once it is published.

---------------
## How to run

Test cases are implemented in ``test.py`` file, which contains experiments for racetrack, elevators and aircraft routing domains.
Please activate which test to execute at the end of ``test.py``.

---------------
## Benchmarks

Benchmark methods, MILP-based method [1] and i-dual [2], are implemented in ``MILPSolver.py`` and ``IDUAL.py``, and there test cases are in ``tests_baselines.py``.
All the test scenarios used for benchmark study are listed in ``models/scenarios``.

[1] D. Dolgov, E. Durfee, Stationary deterministic policies for constrained MDPs with multiple rewards, costs, and discount factors, in: International Joint Conference on Artificial Intelligence, Vol. 19, LAWRENCE ERLBAUM ASSOCIATES LTD, 2005, p. 1326.

[2] F. Trevizan, S. Thiébaux, P. Santana, B. Williams, I-dual: solving constrained SSPs via heuristic search in the dual space, in: Proceedings of the 26th International Joint Conference on Artificial Intelligence, AAAI Press, 2017, pp. 4954–4958.

---------------
## Note

The anytime algorithm was executed with pypy3 and the baseline methods were solved with Gurobi 9. 
