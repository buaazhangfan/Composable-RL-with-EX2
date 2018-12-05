import numpy as np
import tensorflow as tf



# def sample_batch(data, data_size, batch_size):
#     idxs = np.random.randint(data_size, size=batch_size)

#     return data[idxs]

class SimpleSampleLayer():

	def __init__(self, mean, log_var, seed=np.random.randint(1, 2147462579)):
		
		self._srng = np.random.RandomState(seed=seed)
		self.mean = mean
		self.log_var = log_var
		self.seed = seed

	def get_output_for(self):
	
		# eps = self._srng.normal(size=self.mean.shape)
		eps = tf.random_normal(shape = tf.shape(self.mean), seed=self.seed)
		return self.mean + tf.exp(0.5 * self.log_var) * eps


class MLP():
	def __init__(self, input_layer, output_dim, hidden_sizes, hidden_act=tf.tanh, output_act=tf.identity, params=None, batch_norm=False, dropout=False):
		
		out_layer = input_layer
		# param_idx = 0

		for hidden_size in hidden_sizes:

			# Reuse?

			out_layer = tf.layers.dense(out_layer, hidden_size, activation=hidden_act)
			if batch_norm:
				out_layer = tf.layers.batch_normalization(out_layer)
			if dropout:
				out_layer = tf.layers.dropout(out_layer)
		
		if output_act is tf.sigmoid:
			self.before_sig = tf.layers.dense(out_layer, output_dim, activation=tf.identity)
			self.out_layer = tf.sigmoid(self.before_sig)
		else:
			self.out_layer = tf.layers.dense(out_layer, output_dim, activation=output_act)

	def output_layer(self):
		return self.out_layer

class ConvNet():
	def __init__(self, input_layer, filter_sizes=((3,3), (3,3)),
				 number_filters=(16,16),
				 hidden_act=tf.nn.relu):
		
		out_layer = input_layer
		for i, (filter_size, number_filter) in enumerate(zip(filter_sizes, number_filters)):

			out_layer = tf.layers.conv2d(out_layer, filters=number_filter, kernel_size=filter_size)
		
		out_layer = tf.layers.flatten(out_layer)

		self.out_layer = out_layer

	def output_layer(self):
		
		return self.out_layer


class Siamese():
	def __init__(self,input_dim, feature_dim, hidden_sizes,
				 l2_reg=0, hidden_act=tf.tanh, learning_rate=1e-4,
				 kl_weight=1,
				 batch_norm=False,
				 dropout=False):
		
		self.input_dim = input_dim
		self.feature_dim = feature_dim
		self.hidden_sizes = hidden_sizes
		self.hidden_act = hidden_act
		self.learning_rate = learning_rate
		self.batch_norm = batch_norm
		self.dropout = dropout
		self.kl_weight = kl_weight

		self.build_graph()
	
	def build_graph(self):

		# Define Placeholder(input)
		self.lin1 = tf.placeholder(tf.float32, [None, self.input_dim])
		self.lin2 = tf.placeholder(tf.float32, [None, self.input_dim])
		self.label = tf.placeholder(tf.float32, [None, 1])

		self.base1 = MLP(self.lin1, self.hidden_sizes[0], self.hidden_sizes, self.hidden_act, self.hidden_act, batch_norm=self.batch_norm)
		self.base2 = MLP(self.lin2, self.hidden_sizes[0], self.hidden_sizes, self.hidden_act, self.hidden_act, batch_norm=self.batch_norm, dropout=self.dropout)
		
		l1_enc_h2 = self.base1.output_layer()
		l2_enc_h2 = self.base2.output_layer()

		self.mean_net1 = MLP(l1_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act)
		self.mean_net2 = MLP(l2_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act, dropout=self.dropout)

		self.logvar_net1 = MLP(l1_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act)
		self.logvar_net2 = MLP(l2_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act, dropout=self.dropout)

		l1_mu = self.mean_net1.output_layer()
		l1_log_var = self.logvar_net1.output_layer()
 
		l2_mu = self.mean_net2.output_layer()
		l2_log_var = self.logvar_net2.output_layer()

		l1_z = SimpleSampleLayer(mean=l1_mu, log_var=l1_log_var).get_output_for()
		l2_z = SimpleSampleLayer(mean=l2_mu, log_var=l2_log_var).get_output_for()

		

		combined_z = tf.concat([l1_z, l2_z], axis = 1)
		combined_z_mu = tf.concat([l1_mu, l2_mu], axis = 1)
		combined_z_log_var = tf.concat([l1_log_var, l2_log_var], axis = 1)

		self.class_net = MLP(combined_z, 1, self.hidden_sizes, hidden_act=self.hidden_act, output_act=tf.sigmoid)

		self.vae_output = self.class_net.output_layer()

		self.vae_before_sig_output = self.class_net.before_sig

		self.loss = -self.latent_gaussian_x_bernoulli(combined_z, combined_z_mu, combined_z_log_var, self.vae_before_sig_output, self.label, self.kl_weight)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def kl_normal2_stdnormal(self, mean, log_var):

		return - 0.5 * (1 + log_var - mean ** 2 - tf.exp(log_var))

	def log_bernoulli(self, output, label, eps=0.0):

		return - tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)

	def latent_gaussian_x_bernoulli(self, z, z_mu, z_log_var, output, label, kl_weight):
		kl_term = tf.reduce_sum(self.kl_normal2_stdnormal(z_mu, z_log_var),axis = 1)
		log_px_given_z = tf.reduce_sum(self.log_bernoulli(output, label, eps=1e-6), axis = 1)
		Loss = tf.reduce_mean((-kl_term) * kl_weight + log_px_given_z)

		return Loss

	def init_tf_sess(self):
		tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
		tf_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=tf_config)
		self.sess.__enter__()
		tf.global_variables_initializer().run()
	
	def train(self, input_train_1, input_train_2, label_train):
		_, loss = self.sess.run([self.optimizer, self.loss], feed_dict = {self.lin1: input_train_1, self.lin2: input_train_2, self.label: label_train})

		return loss
	
	def predict(self, input_1, input_2):

		dis_output = self.sess.run(self.vae_output, feed_dict = {self.lin1: input_1, self.lin2: input_2})
		dis_output = np.clip(np.squeeze(dis_output), 1e-5, 1-1e-5)
		prob = (1 - dis_output) / (dis_output)

		return prob




class SiameseConv():
	def __init__(self,input_dim, feature_dim, hidden_sizes,
				 l2_reg=0, hidden_act=tf.tanh, learning_rate=1e-4,
				 kl_weight=1,
				 batch_norm=False,
				 dropout=False,
				 img_width,img_height,
				 channel_size=3
				 ):
		
		self.input_dim = input_dim
		self.feature_dim = feature_dim
		self.hidden_sizes = hidden_sizes
		self.hidden_act = hidden_act
		self.learning_rate = learning_rate
		self.batch_norm = batch_norm
		self.dropout = dropout
		self.kl_weight = kl_weight
		self.img_width = img_width
		self.img_height = img_height
		self.channel_size = channel_size

		self.build_graph()
	
	def build_graph(self):

		# Define Placeholder(input)
		self.lin1 = tf.placeholder(tf.float32, [None, self.input_dim])
		self.lin2 = tf.placeholder(tf.float32, [None, self.input_dim])
		self.label = tf.placeholder(tf.float32, [None, 1])

		lin1 = self.lin1
		lin2 = self.lin2

		lin1 = tf.reshape(lin1, shape=[-1, self.channel_size, self.img_width, self.img_height])
		lin2 = tf.reshape(lin2, shape=[-1, self.channel_size, self.img_width, self.img_height])

		self.base1 = ConvNet(lin1)
		self.base2 = ConvNet(lin2)
		# self.base1 = MLP(self.lin1, self.hidden_sizes[0], self.hidden_sizes, self.hidden_act, self.hidden_act, batch_norm=self.batch_norm)
		# self.base2 = MLP(self.lin2, self.hidden_sizes[0], self.hidden_sizes, self.hidden_act, self.hidden_act, batch_norm=self.batch_norm, dropout=self.dropout)
		
		l1_enc_h2 = self.base1.output_layer()
		l2_enc_h2 = self.base2.output_layer()

		self.mean_net1 = MLP(l1_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act)
		self.mean_net2 = MLP(l2_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act, dropout=self.dropout)

		self.logvar_net1 = MLP(l1_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act)
		self.logvar_net2 = MLP(l2_enc_h2, self.feature_dim, self.hidden_sizes, self.hidden_act, dropout=self.dropout)

		l1_mu = self.mean_net1.output_layer()
		l1_log_var = self.logvar_net1.output_layer()
 
		l2_mu = self.mean_net2.output_layer()
		l2_log_var = self.logvar_net2.output_layer()

		l1_z = SimpleSampleLayer(mean=l1_mu, log_var=l1_log_var).get_output_for()
		l2_z = SimpleSampleLayer(mean=l2_mu, log_var=l2_log_var).get_output_for()

		

		combined_z = tf.concat([l1_z, l2_z], axis = 1)
		combined_z_mu = tf.concat([l1_mu, l2_mu], axis = 1)
		combined_z_log_var = tf.concat([l1_log_var, l2_log_var], axis = 1)

		self.class_net = MLP(combined_z, 1, self.hidden_sizes, hidden_act=self.hidden_act, output_act=tf.sigmoid)

		self.vae_output = self.class_net.output_layer()

		self.vae_before_sig_output = self.class_net.before_sig

		self.loss = -self.latent_gaussian_x_bernoulli(combined_z, combined_z_mu, combined_z_log_var, self.vae_before_sig_output, self.label, self.kl_weight)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def kl_normal2_stdnormal(self, mean, log_var):

		return - 0.5 * (1 + log_var - mean ** 2 - tf.exp(log_var))

	def log_bernoulli(self, output, label, eps=0.0):

		return - tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)

	def latent_gaussian_x_bernoulli(self, z, z_mu, z_log_var, output, label, kl_weight):
		kl_term = tf.reduce_sum(self.kl_normal2_stdnormal(z_mu, z_log_var),axis = 1)
		log_px_given_z = tf.reduce_sum(self.log_bernoulli(output, label, eps=1e-6), axis = 1)
		Loss = tf.reduce_mean((-kl_term) * kl_weight + log_px_given_z)

		return Loss

	def init_tf_sess(self):
		tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
		tf_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=tf_config)
		self.sess.__enter__()
		tf.global_variables_initializer().run()
	
	def train(self, input_train_1, input_train_2, label_train):
		_, loss = self.sess.run([self.optimizer, self.loss], feed_dict = {self.lin1: input_train_1, self.lin2: input_train_2, self.label: label_train})

		return loss
	
	def predict(self, input_1, input_2):

		dis_output = self.sess.run(self.vae_output, feed_dict = {self.lin1: input_1, self.lin2: input_2})
		dis_output = np.clip(np.squeeze(dis_output), 1e-5, 1-1e-5)
		prob = (1 - dis_output) / (dis_output)

		return prob

