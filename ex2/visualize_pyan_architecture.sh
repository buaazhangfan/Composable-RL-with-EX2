#!/bin/bash
echo -ne "Pyan architecture: generating architecture.{dot,svg}\n"
/Users/wangyujie/Desktop/iProud/iCourse/US/294-Reinforcement_Learning/viz_code/pyan/pyan.py dqn_utils.py --defines --uses --colored --grouped --annotate --dot -V >dqn_utils.dot 2>dqn_utils.log
dot -Tsvg dqn_utils.dot >dqn_utils.svg
