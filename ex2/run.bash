#!/bin/bash
# python3 run_dqn_lander.py --explore soft_q
python3 run_dqn_lander.py --explore soft_q --ex2 --coef 0.01
python3 run_dqn_lander.py --explore soft_q --ex2 --coef 0.001
python3 run_dqn_lander.py --explore soft_q --ex2 --coef 0.0001
