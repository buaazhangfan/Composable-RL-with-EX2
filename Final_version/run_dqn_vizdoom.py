import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

# VIP: different from other tasks
import dqn_vizdoom as dqn
from dqn_utils import *
import vizdoom as vzd
from tqdm import trange

def vizdoom_model(img_in, num_actions, scope, reuse=False, dropout=False, keep_prob=1.0):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
            out = layers.convolution2d(out, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))           
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=128,                     activation_fn=tf.nn.relu,
                                              weights_initializer=layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None,
                                              weights_initializer=layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))

        return out

def vizdoom_learn(game,
                  session,
                  num_timesteps,
                  double_q,
                  explore,
                  ex2,
                  coef,
                  vizdoom,
                  seed,
                  evaluation):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = ConstantSchedule(2.5e-4)
    # lr_schedule = PiecewiseSchedule([
    #                                      (0,                   1e-4 * lr_multiplier),
    #                                      (num_iterations / 10, 1e-4 * lr_multiplier),
    #                                      (num_iterations / 2,  5e-5 * lr_multiplier),
    #                                 ],
    #                                 outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    # TO-DO: FIGURE OUT HOW TO WRAP UP
    # TO-DO: CHECK MONITOR SETTING
    # ONLY NEED IS TO STOP THE ENVIRONMENT
    # SEEMS THAT WE CAN DEFINE BY OURSELVES
    # TO-DO: see if works
    if vizdoom:
        def stopping_criterion(env, t):
            None
    else:
        def stopping_criterion(env, t):
            # notice that here t is the number of steps of the wrapped env,
            # which is different from the number of steps in the underlying env
            return get_wrapper_by_name(game, "Monitor").get_total_steps() >= num_timesteps
    # therefore, the exploration gradually decrease
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (0.6*8e4, 0.1),
        ], outside_value=0.1
    )

    # TO-DO: Pay attention to arg here, double_q
    dqn.learn(
        env=game,
        q_func=vizdoom_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        # changed for vizdoom
        replay_buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        # Changed for vizdoom
        learning_starts=0,
        learning_freq=1,
        frame_history_len=1,
        target_update_freq=1,
        grad_norm_clipping=10,
        double_q=double_q,
        rew_file='./pkl/vizdoom_'+time.strftime("%d-%m-%Y_%H-%M-%S")+'.pkl',
        explore=explore,
        ex2=ex2,
        ex2_len=128,
        min_replay_size=128,
        coef=coef,
        seed=seed,
        eval= evaluation,
        vizdoom=vizdoom,
        # model_path= './bstmodel/vizdoom_'+time.strftime("%d-%m-%Y_%H-%M-%S")
        model_path = './bstmodel/vizdoom_12-12-2018_01-28-25'
    )
    game.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
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

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path, seed):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    # TO-DO: set seed
    game.set_seed(seed)
    game.init()
    print("Doom initialized.")
    return game

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--vizdoom', action='store_true')
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--explore', type=str, default='e-greedy')
    parser.add_argument('--ex2', action='store_true')
    parser.add_argument('--coef', type=float, default=0.01)
    args = parser.parse_args()
    
    # Run training
    # seed = random.randint(0, 9999)
    seed = 300
    print('random seed = %d' % seed)
    set_global_seeds(seed)

    # Get Vizdoom games.
    # Create Doom instance
    DEFAULT_CONFIG = "/Users/wangyujie/Desktop/iProud/iCourse/US/294-Reinforcement_Learning/Group_Project/ViZDoom/scenarios/simpler_basic.cfg"
    game = initialize_vizdoom(DEFAULT_CONFIG, seed)
    print('using game vizdoom')

    # env = get_env(task, seed, args.env_name)
    session = get_session()
    
    # OMG, 200M Maximum steps
    # TO-DO: num_timesteps need to be changed, here 8e4 = epochs * it_per_epochs
    vizdoom_learn(game, session, num_timesteps=8e4, vizdoom = args.vizdoom, double_q=args.double_q, 
                  explore=args.explore, ex2=args.ex2, coef=args.coef, seed=seed, evaluation=args.eval)

if __name__ == "__main__":
    main()
