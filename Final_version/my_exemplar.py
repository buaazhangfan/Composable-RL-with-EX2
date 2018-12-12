from siamese_tf import Siamese
import numpy as np
import tensorflow as tf


class Exemplar(object):
    def __init__(
            self,
            input_dim,
            seed = 250,
            feature_dim = 4, 
            hidden_sizes = (32,32),
            bonus_form= "1/sqrt(p)",
            eval= False
        ):
        self.first_train = False
        self.bonus_form = bonus_form
        self.model = Siamese(input_dim, feature_dim, hidden_sizes, seed = seed, eval=eval)
        #self.model.init_tf_sess(sess)

    def fit(self, positive, negative):
        #log_step = self.train_itrs * self.log_freq
        #print('pshape', len(positive))
        batch_size = len(positive)
        labels = np.expand_dims(np.concatenate([np.ones(batch_size), np.zeros(batch_size)]), 1).astype(np.float32)
        x1 = np.concatenate([positive, positive])
        x2 = np.concatenate([positive, negative])
        loss = self.model.train(x1, x2, labels)
        return loss

    def predict(self, path):
        counts = self.model.predict(path, path)
        # print(counts)
        # if self.rank == 0:
        #     logger.record_tabular('Average Prob', np.mean(counts))
        #     logger.record_tabular('Average Discrim', np.mean(1/(5.01*counts + 1)))

        if self.bonus_form == "1/n":
            bonuses = 1./counts
        elif self.bonus_form == "1/sqrt(pn)":
            bonuses = 1. / np.sqrt(self.replay.size * counts)
        elif self.bonus_form == "1/sqrt(p)":
            bonuses = 1./np.sqrt(counts)
        elif self.bonus_form == "1/log(n+1)":
            bonuses = 1./np.log(counts + 1)
        elif self.bonus_form == "1/log(n)":
            bonuses = 1. / np.log(counts)
        elif self.bonus_form == "-log(p)":
            bonuses = - np.log(counts)
        else:
            raise NotImplementedError
        return bonuses
