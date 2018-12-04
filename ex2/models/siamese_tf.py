import numpy as numpy
import tensorflow as tensorflow
from ex2.utils.distributions import log_stdnormal, log_normal2, log_bernoulli, kl_normal2_stdnormal
from ex2.utils.theano_utils import compile_timer


NORM_CONSTANT = 2 * np.sqrt(np.pi * 2).item()

def make_mlp(l_in, hidden_sizes, input_dim, 
			 output_dim, hidden_act=tf.tanh, final_act=None):