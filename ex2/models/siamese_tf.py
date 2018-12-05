import numpy as np
import tensorflow as tf
from ex2.utils.distributions import log_stdnormal, log_normal2, log_bernoulli, kl_normal2_stdnormal
from ex2.utils.theano_utils import compile_timer


def sample_batch(data, data_size, batch_size):
    idxs = np.random.randint(data_size, size=batch_size)

    return data[idxs]

class SimpleSampleLayer():

    def __init__(self, mean, log_var, seed=np.random.randint(1, 2147462579), **kwargs):
        self._srng = np.random.RandomState(seed)

    def seed(self, seed=np.random.randint(1, 2147462579)):
        self._srng.seed(seed)
    
    def get_output_for(self, input, deterministic=False):
        mu, log_var = input
        eps = self._srng.normal(mu.shape)
        z = mu + tf.exp(0.5 * log_var) * eps
        if deterministic:
            z = mu

        return z


class MLP():
    def __init__(self, input_layer, output_dim, hidden_sizes,
                 hidden_act=tf.tanh,
                 output_act=tf.identity,
                 params=None,
                 batch_norm=False,
                 dropout=False):
        out_layer = input_layer
        param_idx = 0

        for hidden_size in hidden_sizes:

            # Reuse?

            out_layer = tf.layers.dense(out_layer, hidden_size, activation=hidden_act)
            if batch_norm:
                out_layer = tf.layers.batch_normalization(out_layer)
            if dropout:
                out_layer = tf.layers.dropout(out_layer)
        
        out_layer = tf.layers.dense(out_layer, output_dim, activation=output_act)

        self.out_layer = out_layer
    
    def output_layer(self):
        return self.out_layer

class Siamese:
    def __init__(self, input_dim, feature_dim, hidden_sizes,
                 l2_reg=0, hidden_act=tf.tanh, learning_rate=1e-4,
                 kl_weight=1,
                 batch_norm=False,
                 use_cos=False,
                 dropout=False,
                 env_name='Maze'):
        
        self.input_dim = input_dim
        self.env_name = env_name

        # Define Placeholder(input)
        self.lin1 = tf.placeholder(tf.float64, [None, input_dim])
        self.lin2 = tf.placeholder(tf.float64, [None, input_dim])
        self.label = tf.placeholder(tf.float64, [None, 1])

        self.base1 = MLP(self.lin1, hidden_sizes[0], hidden_sizes,
                         hidden_act, hidden_act, batch_norm=batch_norm)
        self.base2 = MLP(self.lin2, hidden_sizes[0], hidden_sizes,
                         hidden_act, hidden_act, batch_norm=batch_norm,
                         dropout=dropout)
        
        l1_enc_h2 = self.base1.output_layer()
        l2_enc_h2 = self.base2.output_layer()

        self.mean_net1 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)
        self.mean_net2 = MLP(l2_enc_h2, feature_dim, hidden_sizes, hidden_act, dropout=dropout)

        self.logvar_net1 = MLP(l1_enc_h2, feature_dim, hidden_sizes, hidden_act)
        self.logvar_net2 = MLP(l2_enc_h2, feature_dim, hidden_sizes, hidden_act, dropout=dropout)

        l1_mu = self.mean_net1.output_layer()
        l1_log_var = self.logvar_net1.output_layer()
 
        l2_mu = self.mean_net2.output_layer()
        l2_log_var = self.logvar_net2.output_layer()

        l1_z = SimpleSampleLayer(l1_mu, l1_log_var)
        l2_z = SimpleSampleLayer(l2_mu, l2_log_var)

        combined_z = tf.concat([l1_z, l2_z], axis = -1)

        self.class_net = MLP(combined_z, 1, hidden_sizes, hidden_act=hidden_act, output_act=tf.sigmoid)

        self.vae_output = self.class_net.output_layer
