# CS294-112 HW 3: Q-Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.

# Instructions for Assignment3
## Part 1: Q Learning

### Question 1: Basic Q-learning Performance
#### 1. Training

```ruby
python run_dqn_atari.py PongNoFrameskip-v4 --explore e-greedy
```
#### 2. Plot Result

```ruby
python p1_q1.py
```

### Question 2: double Q-learning
#### 1. Training

```ruby
python run_dqn_atari.py PongNoFrameskip-v4 --explore e-greedypython run_dqn_atari.py PongNoFrameskip-v4 --double_q --explore e-greedypython run_dqn_atari.py ZaxxonNoFrameskip-v4 --explore e-greedypython run_dqn_atari.py ZaxxonNoFrameskip-v4 --double_q --explore e-greedy
```
#### 2. Plot Result

```ruby
python p1_q2.pypython p1_q2_z.py
```

### Question 3: experimenting with hyperparameters
#### 1. Training

```ruby
python run_dqn_atari.py PongNoFrameskip-v4 --explore greedypython run_dqn_atari.py PongNoFrameskip-v4 --explore e-greedypython run_dqn_atari.py PongNoFrameskip-v4 --explore boltzmannpython run_dqn_atari.py PongNoFrameskip-v4 --explore bayesian
```
#### 2. Plot Result

```ruby
python p1_q3.py
```
#### 3. Reference for Exploration Strategy
```ruby
[Theory] https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part- 7-action-selection-strategies-for-exploration-d3a97b7cceaf[Code] https://github.com/awjuliani/DeepRL-Agents/blob/master/Q-Exploration.ipynb
```



## Part 2: Actor-Critic

### Question 1: Sanity check with Cartpole
#### 1. Training

```ruby
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3  --exp_name 1_1 -ntu 1 -ngsptu 1python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3  --exp_name 100_1 -ntu 100 -ngsptu 1python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3  --exp_name 1_100 -ntu 1 -ngsptu 100python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3  --exp_name 10_10 -ntu 10 -ngsptu 10
```
#### 2. Plot Result

```ruby
python plot.py data/*CartPole* --value AverageReturn
```

### Question 2: Run actor-critic with more diffcult tasks
#### 1. Training

```ruby
 python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10
 python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10
```
#### 2. Plot Result

```ruby
python plot.py data/*InvertedPendulum* --value AverageReturnpython plot.py data/*HalfCheetah* --value AverageReturn
```



## Cheers!