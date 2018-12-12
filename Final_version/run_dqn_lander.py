import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

import dqn
from dqn_utils import *

def lander_model(obs, num_actions, scope, reuse=False, dropout=False, keep_prob=1.0):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def lander_optimizer():
    return dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )

def lander_stopping_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    return stopping_criterion

def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def lander_kwargs():
    return {
        'optimizer_spec': lander_optimizer(),
        'q_func': lander_model,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 1.00,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }

def lander_learn(env,
                 session,
                 num_timesteps,
                 seed,
                 double_q,
                 explore,
                 ex2,
                 coef
                 ):

    optimizer = lander_optimizer()
    stopping_criterion = lander_stopping_criterion(num_timesteps)
    exploration_schedule = lander_exploration_schedule(num_timesteps)

    dqn.learn(
        env=env,
        session=session,
        exploration=lander_exploration_schedule(num_timesteps),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        # double_q=True,
        double_q = double_q,
        rew_file='./pkl/lander_'+ time.strftime("%d-%m-%Y_%H-%M-%S") +'.pkl',
        explore=explore,
        ex2=ex2,
        coef=coef,
        seed=seed,
        **lander_kwargs()
    )
    env.close()

def lander_eval(env,
                 session,
                 num_timesteps,
                 seed,
                 double_q,
                 explore,
                 ex2,
                 coef
                 ):

    optimizer = lander_optimizer()
    stopping_criterion = lander_stopping_criterion(num_timesteps)
    exploration_schedule = lander_exploration_schedule(num_timesteps)

    dqn.evaluate(
        env=env,
        session=session,
        exploration=lander_exploration_schedule(num_timesteps),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        # double_q=True,
        double_q = double_q,
        rew_file='./pkl/lander_'+ time.strftime("%d-%m-%Y_%H-%M-%S") +'.pkl',
        explore=explore,
        ex2=ex2,
        coef=coef,
        seed=seed,
        eval=True,
        **lander_kwargs()
    )
    env.close()

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count={'GPU': 0})
    # GPUs don't significantly speed up deep Q-learning for lunar lander,
    # since the observations are low-dimensional
    session = tf.Session(config=tf_config)
    return session

def get_env(seed):
    env = gym.make('LunarLander-v2')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True, video_callable=False)
    # env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    return env

seed = 300 # you may want to randomize this

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--explore', type=str, default='e-greedy')
    parser.add_argument('--ex2', action='store_true')
    parser.add_argument('--coef', type=float, default=0.01)
    args = parser.parse_args()

    # Run training
    # seed = 250 # you may want to randomize this
    print('random seed = %d' % seed)
    env = get_env(seed)
    session = get_session()
    set_global_seeds(seed)
    # lander_learn(env, session, num_timesteps=500000, seed=seed, 
    #            double_q=args.double_q, explore=args.explore, ex2=args.ex2, coef=args.coef)
    lander_eval(env, session, num_timesteps=500000, seed=seed, 
                 double_q=args.double_q, explore=args.explore, ex2=args.ex2, coef=args.coef)
if __name__ == "__main__":
    main()
