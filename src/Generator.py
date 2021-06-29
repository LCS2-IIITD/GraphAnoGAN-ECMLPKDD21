# coding=utf-8
from Model import *


class Generator(GCN):

    def __init__(self, placeholders, input_dim, learning_rate=1e-2):
        GCN.__init__(self, placeholders, input_dim, logging=True)
        self.model_type = "Gen"

        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.index = tf.placeholder(tf.int32, shape=[None], name='index')

        self.batch_scores = tf.nn.softmax(self.score)
        self.prob = tf.gather(self.batch_scores, self.index)
        self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # minimize attention
        self.gan_score = self.score
