# CS294-112 Final Project
### Bridging Distribution Mismatch: Better Bound on Composable Deep Reinforcement Learning


Dependencies:

 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg
 * Vizdoom

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The code was based on an implementation of Q-learning in HW3.

This code can support Atari games and Vizdoom environments.

# Instructions for Composable Soft Q Learning

```ruby
python run_dqn_vizdoom.py --vizdoom --explore e-greedy --subgame shoot_monster
python run_dqn_vizdoom.py --vizdoom --explore soft_q --subgame avoid_shooters
python run_dqn_vizdoom.py --vizdoom --explore soft_q --ex2 --coef 1e-4 --subgame avoid_shooters
```

## Cheers!