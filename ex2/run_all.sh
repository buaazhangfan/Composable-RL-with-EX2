#!/usr/bin/env bash
python run_dqn_vizdoom.py --vizdoom --explore e-greedy --subgame shoot_monster
python run_dqn_vizdoom.py --vizdoom --explore soft_q --subgame shoot_monster
python run_dqn_vizdoom.py --vizdoom --explore soft_q --ex2 --coef 1e-2 --subgame shoot_monster


python run_dqn_vizdoom.py --vizdoom --explore e-greedy --subgame avoid_shooters
python run_dqn_vizdoom.py --vizdoom --explore soft_q --subgame avoid_shooters
python run_dqn_vizdoom.py --vizdoom --explore soft_q --ex2 --coef 1e-2 --subgame avoid_shooters


python run_dqn_vizdoom.py --vizdoom --explore e-greedy --subgame avoid+shoot
python run_dqn_vizdoom.py --vizdoom --explore soft_q --subgame avoid+shoot
python run_dqn_vizdoom.py --vizdoom --explore soft_q --ex2 --coef 1e-2 --subgame avoid+shoot