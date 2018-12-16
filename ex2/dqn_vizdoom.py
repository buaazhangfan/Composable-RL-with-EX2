import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
from my_exemplar_conv import Exemplar

import skimage.color, skimage.transform
import itertools as it
from tqdm import trange
import logging

frame_repeat = 12
resolution = (30, 45)
epochs = 80
learning_steps_per_epoch = 2000
test_episodes_per_epoch = 100

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img  = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32)
    return img

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):
  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False,
    explore='e-greedy',
    ex2= False,
    min_replay_size=10,
    # TO-DO: not sure
    ex2_len= 10,
    coef=0.01,
    seed=250,
    eval=False,
    vizdoom=False,
    model_path=None,
    subgame=None):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    # Different setting for gym and vizdoom
    self.vizdoom = vizdoom
    self.model_path = model_path
    self.subgame = subgame

    if not self.vizdoom:
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
    self.double_q = double_q
    self.explore = explore
    # EX2
    # [1e-3, 1e-4, 1e-5]
    self.coef = coef
    self.first_train = True
    self.first_train_itrs = int(5e3)
    self.train_itrs = int(1e3)
    self.ex2 = ex2
    self.min_replay_size = min_replay_size
    self.ex2_len = ex2_len
    self.count = 0
    self.seed = seed
    self.eval = eval
    self.log_name = str(uuid.uuid4()) + '_' + self.explore + 'ex2_' + str(self.ex2) + 'coef_' + str(self.coef) + '.log'
    print('eval?', self.eval)
    print('exploration strategy', explore)
    print('using ex2', ex2)
    print('using coef', coef)
    set_global_seeds(seed)
    print('seed set again')
    
    ###############
    # BUILD MODEL #
    ###############
    if self.vizdoom:
        channels = 1
        input_shape = (resolution[0], resolution[1], channels)
        # Action = which buttons are pressed
        n = self.env.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.num_actions = len(self.actions)
    else:
        if len(self.env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            # IT is what I am debugging on!
            input_shape = self.env.observation_space.shape
        else:
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
        self.num_actions = self.env.action_space.n
    
    if self.eval:
        # saver1 = tf.train.import_meta_graph('./bstmodel/model.meta')
        # saver1.restore(self.session, './bstmodel/model')
        saver1 = tf.train.import_meta_graph(self.model_path+'.meta')
        saver1.restore(self.session, self.model_path)
        self.obs_t_ph = tf.get_collection('obs_t_ph')[0]
        if self.explore == 'soft_q':
            self.Temp = tf.get_collection('Temp')[0]
            self.q_dist = tf.get_collection('q_dist')[0]
        self.keep_per = tf.get_collection('keep_per')[0]
        
        self.q_t = tf.get_collection('q_t')[0]
        # Ex2
        if self.ex2:
            self.ex2_in1 = tf.get_collection('ex2_in1')[0]
            self.ex2_in2 = tf.get_collection('ex2_in2')[0]
            self.ex2_dis_output = tf.get_collection('ex2_dis_output')[0]
            self.ex2_prob = tf.get_collection('ex2_prob')[0]
        self.model_initialized = True
        # print('obs is here',self.obs_t_ph)
        # print(self.Temp)
        # print(self.keep_per)
        # print(self.q_dist)

        print('restored and initialized the model')
    else:
        # set up placeholders
        # placeholder for current observation (or state)
        self.obs_t_ph              = tf.placeholder(
            tf.float32 if (lander or vizdoom) else tf.uint8, [None] + list(input_shape))
        # placeholder for current action
        self.act_t_ph              = tf.placeholder(tf.int32,   [None])
        # placeholder for current reward
        self.rew_t_ph              = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph            = tf.placeholder(
            tf.float32 if (lander or vizdoom) else tf.uint8, [None] + list(input_shape))
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph          = tf.placeholder(tf.float32, [None])

        # casting to float on GPU ensures lower data transfer times.
        # TO-DO: WHY?
        if (lander or vizdoom):
          obs_t_float = self.obs_t_ph
          obs_tp1_float = self.obs_tp1_ph
        else:
          obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
          obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        # Here, you should fill in your own code to compute the Bellman error. This requires
        # evaluating the current and next Q-values and constructing the corresponding error.
        # TensorFlow will differentiate this error for you, you just need to pass it to the
        # optimizer. See assignment text for details.

        # Your code should produce one scalar-valued tensor: total_error
        # This will be passed to the optimizer in the provided code below.

        # Your code should also produce two collections of variables:
        # q_func_vars
        # target_q_func_vars
        # These should hold all of the variables of the Q-function network and target network,
        # respectively. A convenient way to get these is to make use of TF's "scope" feature.
        # For example, you can create your Q-function network with the scope "q_func" like this:
        # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        # And then you can obtain the variables like this:
        # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')   
        # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"

        # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
        ######

        # YOUR CODE HERE
        # For bayesian exploration: Add dropout value to the network 
        # Get Q-function and target network
        self.keep_per = tf.placeholder(shape=None,dtype=tf.float32)
        if self.explore == 'bayesian':
            print('Bayesian variables defined!')
            dropout = True
        else:
            dropout = False

        # EX2
        if self.ex2:
            print('Use Exemplar Model')
            self.exemplar = Exemplar(input_dim= input_shape[0], seed=self.seed, eval=self.eval )

        q_t   = q_func(obs_t_float, self.num_actions, scope='q_func', reuse=False, 
                           dropout=dropout, keep_prob=self.keep_per)
        q_tp1 = q_func(obs_tp1_float, self.num_actions, scope='target_q_func_vars', reuse=False, 
                           dropout=dropout, keep_prob=self.keep_per)

        # For boltzmann exploration
        if self.explore == 'soft_q':
            print('Boltzman variables defined!')
            self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
            # print(q_t)

            #value = tf.reduce_mean(q_t, 1)
            # print(value)

            # print(q_t - value)
            # print(self.q_dist)
            # exit()
            # self.q_dist = tf.nn.softmax(q_t/self.Temp)

            # # Old version
            # value = tf.log( tf.reduce_sum(tf.exp(q_t),1) )
            # self.q_dist = tf.exp(q_t - value)

            # New version
            self.q_dist = tf.nn.softmax(q_t/self.Temp)

        # Max operation
        self.q_t_action = tf.argmax(q_t, axis=1)
        # value = tf.reduce_mean(q_t)
        # self.q_t_action = tf.nn.softmax(q_t - value)

        # Specify double Q function difference
        if self.double_q:
            print('using double q learning')
            # TO-DO: VERY VERY IMPORTANT TO REUSE VAIRABLES
            # TO-DO: DO WE NEED TO SET GRADIENT NOT UPDATE
            q_tp1_target = q_func(obs_tp1_float, self.num_actions, scope='q_func', reuse=True)
            q_tp1_target_action = tf.argmax(q_tp1_target, axis=1)
            q_tp1_max = tf.reduce_sum(q_tp1 * tf.one_hot(indices=q_tp1_target_action,
                                                         depth=self.num_actions, 
                                                         on_value=1.0, off_value=0.0), axis=1)
        else:
            # Soft maximum
            if self.explore == 'soft_q':
                print('using soft q learning')
                # q_tp1_max = tf.log( tf.reduce_sum(tf.exp(q_tp1),1) )
                q_tp1_max = tf.reduce_logsumexp(q_tp1, 1)
                # print(q_tp1_max)
                # exit()
            else:
                q_tp1_max = tf.reduce_max(q_tp1, 1)

        # Get target value
        q_tp1 = gamma * (1.0 - self.done_mask_ph) * q_tp1_max 
        target = self.rew_t_ph + q_tp1
        # Get Q_fai(si,ai)
        # TO-DO: VERY VERY IMPORTANT! use reduce_sum instead of reduce_max since exist negative value
        q_t_target = tf.reduce_sum(q_t * tf.one_hot(indices=self.act_t_ph, 
                                                    depth=self.num_actions, 
                                                    on_value=1.0, off_value=0.0), axis=1)   

        # Calculate loss
        self.total_error = target - q_t_target
        self.total_error = tf.reduce_mean(huber_loss(self.total_error))

        # Produce collections of variables to update separately
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func') 
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func_vars') 
        if 0:
            print(q_t.get_shape())
            print(q_tp1.get_shape())
            print(self.q_t_action.get_shape())
            print(self.done_mask_ph.get_shape())
            print(q_tp1_max.get_shape())
            print(q_tp1.get_shape())
            print(q_t_target.get_shape())
            print(self.total_error.get_shape())
            exit()
        ######

        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                     var_list=q_func_vars, clip_val=grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    if not self.eval:
        self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    # last_obs intialized here
    if self.vizdoom:
        self.last_obs = None
    else:
        self.last_obs = self.env.reset()
    if self.vizdoom:
        self.log_every_n_steps = learning_steps_per_epoch
    else:
        self.log_every_n_steps = 10000
    self.timesteps = []
    self.mean_episode_rewards = []
    self.best_mean_episode_rewards = []

    self.start_time = None
    self.t = 0

    # EX2
    if not eval:
        self.saver = tf.train.Saver()
        tf.add_to_collection('obs_t_ph', self.obs_t_ph)
        if self.explore == 'soft_q':
            tf.add_to_collection('Temp', self.Temp)
            tf.add_to_collection('q_dist', self.q_dist)
        tf.add_to_collection('keep_per', self.keep_per)
        tf.add_to_collection('q_t', q_t)
        if self.ex2:
            in1, in2, dis_output, prob = self.exemplar.model.predict_tensor()
            tf.add_to_collection('ex2_in1', in1)
            tf.add_to_collection('ex2_in2', in2)
            tf.add_to_collection('ex2_dis_output', dis_output)
            tf.add_to_collection('ex2_prob', prob)


    if self.ex2 and not self.eval:
        self.exemplar.model.init_tf_sess(self.session)
        self.model_initialized = True
    """
    # eval
    if self.eval:
        print("Initialize Evaluation Mode")
        self.saver.restore(self.session, "./bstmodel/model.ckpt")
        self.model_initialized = True
        print("Initialized models")
    """
  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.

    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.

    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!

    ## what do you mean?? what is context?
    ## i think it is just {s,a,r}
    ## check encode_recent_observation!

    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.

    # Don't forget to include epsilon greedy exploration!

    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)
    
    # Ex2
    self.count += 1
    # Store observation
    if self.vizdoom:
        self.last_obs = preprocess(self.env.get_state().screen_buffer)
    ret = self.replay_buffer.store_frame(self.last_obs)
    self.e_current_idx = ret 
    # print(np.shape(self.last_obs))
    
    # For exploration, the value will gradually decrease
    if self.explore == 'greedy':
        # print("using greedy exploration!")
        if (not self.model_initialized):
            action = np.random.randint(0, self.num_actions)
        else:
            recent_obs = self.replay_buffer.encode_recent_observation()
            action = self.session.run(self.q_t_action, feed_dict={self.obs_t_ph: [recent_obs],
                                                                  self.keep_per: 1.0})
            action = action[0]
    if self.explore == 'e-greedy':
        # print("using e-greedy exploration!")
        # Random.random return [0,1)
        if (not self.model_initialized) or (random.random() < self.exploration.value(self.t)):
            action = np.random.randint(0, self.num_actions)
        else:
            # Understanding: context have at least two frames to encode velocity info
            # RECENT_OBS: FOR RAM (128,) AND FOR LAUDER (9,) AND FOR ATARI (84,84,4)
            # Action shape (1,)
            # Encode recent observation
            recent_obs = self.replay_buffer.encode_recent_observation()
            # print(np.shape(recent_obs))
            action = self.session.run(self.q_t_action, feed_dict={self.obs_t_ph: [recent_obs],
                                                                  self.keep_per: 1.0})
            action = action[0]
            # print(np.shape(action))
            # exit()
    if self.explore == 'soft_q':
        # print("using boltzmann exploration!")
        if (not self.model_initialized):
            action = np.random.randint(0, self.num_actions)
        else:
            recent_obs = self.replay_buffer.encode_recent_observation()
            #print(recent_obs.shape)
            #print(recent_obs)
            #print(self.q_dist)
            #print(self.obs_t_ph)
            #print(self.Temp)
            #print(self.keep_per)
            #exit()
            q_d = self.session.run(self.q_dist, feed_dict={self.obs_t_ph: [recent_obs], 
                                                           self.Temp: self.exploration.value(self.t),
                                                           self.keep_per: 1.0})
            if self.eval and (self.replay_buffer.num_in_buffer > self.min_replay_size) and (self.count >= self.ex2_len):
                self.count = 0
                #paths = self.replay_buffer.get_all_positive(self.ex2_len)
                #ex2_out, ex2_pb = self.session.run([self.ex2_dis_output, self.ex2_prob], feed_dict={self.ex2_in1: paths, 
                #                                                       self.ex2_in2: paths})
                
                # print("ex2 dis_out", ex2_out)
                # print("ex2 pb_out", ex2_pb)
                for _ in range(10):
                   ex2_out, ex2_pb = self.session.run([self.ex2_dis_output, self.ex2_prob], feed_dict={self.ex2_in1: [recent_obs], 
                                                                          self.ex2_in2: [recent_obs] })
                   print("ex2_pb", ex2_pb)
            #exit()
            if 0:
                print('in',input_q)
                print('qt',q_t)
                print('qd',q_d)
            action = np.random.choice(self.num_actions, p=q_d[0])

    if self.explore == 'bayesian':
        # print("using bayesian exploration!")
        if (not self.model_initialized):
            action = np.random.randint(0, self.num_actions)
        else:
            recent_obs = self.replay_buffer.encode_recent_observation()
            keep_per = (1.0 - self.exploration.value(self.t)) + 0.1
            # Deal with larger than 1.0 case
            keep_per = 1.0 if keep_per>1.0 else keep_per
            # print(keep_per)
            action = self.session.run(self.q_t_action, feed_dict={self.obs_t_ph: [recent_obs], 
                                                                  self.keep_per: keep_per})
            action = action[0]
            # print(action)
            # exit()
    
    # Step one step forward
    # INPUT FOR ACTION IS INT VALUE
    # what is frame repeat??
    if self.vizdoom:
        reward = self.env.make_action(self.actions[action], frame_repeat)
        done = self.env.is_episode_finished()
        obs = preprocess(self.env.get_state().screen_buffer) if not done else None
    else:
        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()

    # Point to the newest observation
    self.last_obs = obs
    # Store others
    self.replay_buffer.store_effect(ret, action, reward, done)
    
    # Update EX2 model and rewards
    if not self.eval:
        if self.ex2 and (self.replay_buffer.num_in_buffer > self.min_replay_size) and (self.count >= self.ex2_len):
            self.count = 0
            # fit ex2 model
            if self.first_train:
                train_itrs = self.first_train_itrs
                self.first_train = False
            else:
                train_itrs = self.train_itrs
            for i  in range(train_itrs):
                positive = self.replay_buffer.sample_positive(self.ex2_len, 32)
                negative = self.replay_buffer.sample_negative(self.ex2_len, 32)
                # positive_np = np.asarray(positive)
                # print(positive_np.shape)
                # print(self.replay_buffer.num_in_buffer)
                # print(positive)
                # print(len(positive))
                # exit()
                self.exemplar.fit(positive, negative)
                # print("%d in %d" %(i, train_itrs))
            # update rewards
            paths = self.replay_buffer.get_all_positive(self.ex2_len)
            bonus_reward = self.exemplar.predict(paths)
            self.replay_buffer.update_reward(self.ex2_len, bonus_reward, self.coef)
            
    if self.eval:
        self.t += 1
    # exit()
    #####
    # YOUR CODE HERE

  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # Absolutely, this process takes long!
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # batch_size = 32, observation shape = 128
      obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(self.batch_size)

      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!

      # TO-DO: is it only initialize once when first start
      if not self.model_initialized:
          # print("initializing model")
          if self.ex2:
            # initialized in Siamese model
            print("Ex2 no need to initialize")
            pass
          else:
            # print("interdependent init")
            # initialize_interdependent_variables(self.session, tf.global_variables(), {
            #     self.obs_t_ph: obs_t_batch,
            #     self.obs_tp1_ph: obs_tp1_batch,
            # })
            self.session.run(tf.global_variables_initializer())
          # TO-DO: VERY VERY IMPORTATNT!!
          #self.saver = tf.train.Saver()
          # print("set model_initialized True")
          self.model_initialized = True

      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # TO-DO: check written rule okay?
      _, error = self.session.run([self.train_fn, self.total_error], feed_dict={
          self.obs_t_ph: obs_t_batch, 
          self.act_t_ph: act_batch, 
          self.rew_t_ph: rew_batch,
          self.obs_tp1_ph: obs_tp1_batch,
          self.done_mask_ph: done_mask,
          self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t),
          self.keep_per: 1.0})
      # print('error', error)
      # exit()
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####
      # YOUR CODE HERE
      self.num_param_updates += 1
      if (self.num_param_updates % self.target_update_freq == 0):
          # print("actually update")
          self.session.run(self.update_target_fn)
      # exit()
    
    self.t += 1
    # print('self.t', self.t)

  def log_progress(self):
    if self.vizdoom:
        # TO-DO: THIS PART NEED TO BE CHANGED!
        episode_rewards = self.env.get_total_reward()
        self.mean_episode_reward = episode_rewards
        self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
            print("Timestep %d" % (self.t,))
            print("mean reward (100 episodes) %f" % self.mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)
            # print("episodes %d" % len(episode_rewards))
            print("exploration %f" % self.exploration.value(self.t))
            print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
            if self.start_time is not None:
                print("running time %f" % ((time.time() - self.start_time) / 60.))
    else:
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

        if len(episode_rewards) > 0:
          self.mean_episode_reward = np.mean(episode_rewards[-100:])

        if len(episode_rewards) > 50:
          if self.mean_episode_reward > self.best_mean_episode_reward:
            # store the best_mean_reward
            self.best_mean_episode_reward = self.mean_episode_reward
            #print("init?",self.model_initialized)
            #print("eval?",self.eval)
            if self.model_initialized and not self.eval:
                # store the best model
                save_path = self.saver.save(self.session, self.model_path)
                print("Model saved in path: %s" % save_path)
                #self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
                #print("Exemplar test output")
                #for _ in range(10):
                #    self.exemplar.predict(np.ones((1,9))/9 )

        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
          print("Timestep %d" % (self.t,))
          print("mean reward (100 episodes) %f" % self.mean_episode_reward)
          print("best mean reward %f" % self.best_mean_episode_reward)
          print("episodes %d" % len(episode_rewards))
          print("exploration %f" % self.exploration.value(self.t))
          print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
          if self.start_time is not None:
            print("running time %f" % ((time.time() - self.start_time) / 60.))

    self.start_time = time.time()

    sys.stdout.flush()

    # Store variables
    self.timesteps.append(self.t)
    self.mean_episode_rewards.append(self.mean_episode_reward)
    self.best_mean_episode_rewards.append(self.best_mean_episode_reward)

    # TO-DO!!! CHANGE TO FOR LOOP
    # TO-DO: it is weird, since every time it is doing dumpying, but every time it opens as new..
    # Actually if less steps required, we can only store once at the end
    with open(self.rew_file, 'wb') as f:
      store_result = {'timestep': np.array(self.timesteps), 'reward': np.array(episode_rewards), 
                      'mean_reward': np.array(self.mean_episode_rewards), 'best_reward': np.array(self.best_mean_episode_rewards)}
      pickle.dump(store_result, f, pickle.HIGHEST_PROTOCOL)

  def train_test(self):
      best_score = -float('inf')
      for epoch in range(epochs):
          print("\nEpoch %d\n-------" % (epoch + 1))
          train_episodes_finished = 0
          train_scores = []
          logging.basicConfig(filename=self.log_name, level=logging.INFO)
          logging.info("Current epoch is {}".format(epoch))
          print("Training...")
          self.env.new_episode()
          for learning_step in trange(learning_steps_per_epoch, leave=False):
              self.step_env()
              if not self.eval:
                  self.update_model()
              if self.env.is_episode_finished():
                  # print('ended')
                  score = self.env.get_total_reward()
                  train_scores.append(score)
                  self.env.new_episode()
                  train_episodes_finished += 1

                  if not self.subgame == 'shoot_monster':
                      if score > best_score:
                          best_score = score
                          if self.model_initialized and not self.eval:
                              # store the best model
                              # save_path = self.saver.save(self.session, "./bstmodel/Vizdoom_model")
                              save_path = self.saver.save(self.session, self.model_path)
                              print("Model saved in path: %s" % save_path)
          # Add diversity for shoot_monster since it only has a monster to kill, so max is 300.
          if self.subgame == 'shoot_monster':
              if not self.eval:
                    save_path = self.saver.save(self.session, self.model_path)
                    print("Model saved in path: %s" % save_path)
          print("%d training episodes played." % train_episodes_finished)
          train_scores = np.array(train_scores)
          logging.basicConfig(filename='myapp.log', level=logging.INFO)
          logging.info("Results mean: {} and Result std: {}".format(train_scores.mean(), train_scores.std()))
          print("Results: mean: %.1f +- %.1f," % (train_scores.mean(), train_scores.std()), \
                "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
          print("best_score", best_score)
          if 0:
              print("\nTesting...")
              test_episode = []
              test_scores = []
              for test_episode in trange(test_episodes_per_epoch, leave=False):
                  self.env.new_episode()
                  while not self.env.is_episode_finished():
                      state = preprocess(self.env.get_state().screen_buffer)
                      action = self.session.run(self.q_t_action, feed_dict={self.obs_t_ph: [state],
                                                                      self.keep_per: 1.0})
                      action = action[0]
                      # best_action_index = get_best_action(state)
                      self.env.make_action(self.actions[action], frame_repeat)
                      # print(best_action_index)
                  r = self.env.get_total_reward()
                  test_scores.append(r)

              test_scores = np.array(test_scores)
              print("Results: mean: %.1f±%.1f," % (
                  test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                    "max: %.1f" % test_scores.max())


def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  if alg.vizdoom:
      alg.train_test()
          
  else:
      while not alg.stopping_criterion_met():
        alg.step_env()
        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and self.last_obs should point to the new latest
        # observation
        alg.update_model()
        alg.log_progress()
        # print('end')
        # exit()

def evaluate(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    #alg.update_model()
    alg.log_progress()
    # print('end')
    # exit()

