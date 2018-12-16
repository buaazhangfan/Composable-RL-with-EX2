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
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False, dropout=False, keep_prob=1.0):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            if dropout:
                out = layers.dropout(out, keep_prob=keep_prob)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                num_timesteps,
                double_q,
                explore,
                env_name,
                ex2=ex2,
                coef=coef):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    # therefore, the exploration gradually decrease
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    # TO-DO: Pay attention to arg here, double_q
    dqn.learn(
        env=env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=double_q,
        rew_file='./pkl/'+env_name+'_'+time.strftime("%d-%m-%Y_%H-%M-%S")+'.pkl',
        explore=explore,
        ex2=ex2,
        coef=coef
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed, env_name):
    # print(env_name)
    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    # TO-DO: SEE THE VIDEO 
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True, video_callable=False)
    env = wrap_deepmind(env)

    return env

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--explore', type=str, default='e-greedy')
    parser.add_argument('--ex2', action='store_true')
    parser.add_argument('--coef', type=float, default=0.01)
    args = parser.parse_args()
    # Get Atari games.
    task = gym.make(args.env_name)
    print('using env', args.env_name)

    # Run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    env = get_env(task, seed, args.env_name)
    session = get_session()
    # OMG, 200M Maximum steps
    atari_learn(env, session, num_timesteps=2e8, double_q=args.double_q, 
                explore=args.explore, env_name=args.env_name, ex2=args.ex2, coef=args.coef)

if __name__ == "__main__":
    main()
