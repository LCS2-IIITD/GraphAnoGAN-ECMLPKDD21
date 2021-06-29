# coding=utf-8
from Model import *


class Discriminator(GCN):

    def __init__(self, placeholders, input_dim, learning_rate=1e-2):
        GCN.__init__(self, placeholders, input_dim, logging=True)

        self.model_type = "Dis"

        with tf.name_scope("output"):
            self.losses = tf.maximum(0.0, tf.subtract(0.05, self.score))
            self.loss = tf.reduce_sum(self.losses)

            self.reward = 2.0 * (tf.sigmoid(tf.subtract(0.05, self.score)) - 0.5)

            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
