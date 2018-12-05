import numpy as np
import tensorflow as tf



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
        eps = self._srng.normal(size = mu.shape)
        z = mu + tf.exp(0.5 * log_var) * eps
        if deterministic:
            z = mu

        return z


class MLP():
    def __init__(self, input_layer, output_dim, hidden_sizes, hidden_act=tf.tanh, output_act=tf.identity, params=None, batch_norm=False, dropout=False):
		
		out_layer = input_layer
		param_idx = 0

        for hidden_size in hidden_sizes:

            # Reuse?

            out_layer = tf.layers.dense(out_layer, hidden_size, activation=hidden_act)
            if batch_norm:
                out_layer = tf.layers.batch_normalization(out_layer)
            if dropout:
                out_layer = tf.layers.dropout(out_layer)

		self.before_sig_layer = out_layer
        out_layer = tf.layers.dense(out_layer, output_dim, activation=output_act)

        self.out_layer = out_layer

    def before_sig_layer(self):
		return self.before_sig_layer

    def output_layer(self):
        return self.out_layer


class Siamese:
    def __init__(self,input_dim, feature_dim, hidden_sizes,
                 l2_reg=0, hidden_act=tf.tanh, learning_rate=1e-4,
                 kl_weight=1,
                 batch_norm=False,
                 use_cos=False,
                 dropout=False):
        
        self.input_dim = input_dim
        self.env_name = env_name
		self.learning_rate = learning_rate

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

        combined_z = tf.concat([l1_z, l2_z], axis = 1)
		combined_z_mu = tf.concat([l1_mu, l2_mu], axis = 1)
		combined_z_log_var = tf.concat([l1_log_var, l2_log_var], axis = 1)

        self.class_net = MLP(combined_z, 1, hidden_sizes, hidden_act=hidden_act, output_act=tf.sigmoid)

        self.vae_output = self.class_net.output_layer()
		self.vae_before_sig_output = self.class_net.before_sig_layer()

		self.loss = self.latent_gaussian_x_bernoulli(combined_z, combined_z_mu, combined_z_log_var, self.vae_before_sig_output, self.label, kl_weight)
		self.loss *= -1
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def kl_normal2_stdnormal(mean, log_var):

		return -0.5 * (1 + log_var - mean ** 2 - tf.exp(log_var))

	def log_bernoulli(output, label, eps=0.0):

		return - tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)

	def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, output, label, kl_weight):
		kl_term = tf.reduce_sum(self.kl_normal2_stdnormal(z_mu, z_log_var),axis = 1)
		log_px_given_z = tf.reduce_sum(self.log_bernoulli(output, label, eps=1e-6), axis = 1)
		Loss = tf.reduce_mean((-kl_term) * kl_weight + log_px_given_z)

		return Loss

	def init_tf_sess(self):
		tf.config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
		tf_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=tf_config)
		self.sess.__enter__()
		tf.global_variables_initializer().run()
	
	def train(self, input_train_1, input_train_2, label_train):
		_, loss = self.sess.run([self.optimizer, self.loss], feed_dict = {self.lin1 = input_train_1, self.lin2 = input_train_2, self.label = label_train})

		return loss
	
	def predict(self, input_1, input_2):

		dis_output = self.sess.run(self.vae_output, feed_dict = {self.lin1 = input_1, self.lin2 = input_2})

		return dis_output



	

