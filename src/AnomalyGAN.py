# coding=utf-8
# ! /usr/bin/env python3.4
import datetime

from Utils import *
from Generator import *
from Discriminator import *
import sys
temp = sys.stdout
sys.stdout = sys.stderr

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.05, "learning_rate (default: 0.1)")

# data parameters
tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("pools_size", 10, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("g_epochs_num", 1, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 1, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_temperature", 20, " the temperature of sampling")
tf.flags.DEFINE_integer("number_of_vertices", 5, "he number of samples of gan")
tf.flags.DEFINE_integer("gan_k", 10, "he number of samples of gan")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

y_train = []

print(("Loading data..."))
adj, feature, y_train = read_raw()
print("Loading done...")

# Some preprocessing
featureList = []
supportList = []
for featureElement in feature:
    features = preprocess_features(feature[featureElement])
    featureList.append(features)
for adjElement in adj:
    support = preprocess_adj(adj[adjElement])
    supportList.append(support)
num_supports = 1
num_labels = FLAGS.pools_size
dimensionInput = features[2][1]

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': [tf.placeholder(tf.float32) for _ in range(num_labels)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'numberOfChosenGraph': tf.placeholder(tf.int32)
}

loss_precision = '../log/loss.txt'


def generate_gan(sess, model):
    graphListIndex = getListOfGraphs()
    sampled_index = list(graphListIndex)
    numberOfChosenGraph = FLAGS.gan_k

    features, support, yTrain = loadCandidateSamples(sampled_index, featureList, supportList, y_train)
    feed_dict = construct_feed_dict(features, support, yTrain, numberOfChosenGraph, placeholders)

    predicted = sess.run(model.gan_score, feed_dict)

    exp_rating = np.exp(np.array(predicted) * FLAGS.sampled_temperature)
    prob = exp_rating / np.sum(exp_rating)

    genIndex = np.random.choice(sampled_index, size=FLAGS.gan_k, p=prob, replace=False)

    return genIndex


def main():
    with tf.device("/gpu:1"):
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default(), open(loss_precision, "w") as loss_log:
            discriminator = Discriminator(
                placeholders=placeholders,
                input_dim=dimensionInput)

            generator = Generator(
                placeholders=placeholders,
                input_dim=dimensionInput)

            sess.run(tf.global_variables_initializer())
            for i in range(FLAGS.num_epochs):
                if i > 0:
                    graphListIndex = getListOfGraphs()

                    features, support, yTrain = loadCandidateSamples(graphListIndex, featureList, supportList, y_train)
                    numberOfChosenGraph = FLAGS.gan_k

                    feed_dict = construct_feed_dict(features[0], support, yTrain, numberOfChosenGraph, placeholders)
                    _, step, current_loss, accuracy = sess.run(
                        [discriminator.train_op, discriminator.global_step, discriminator.loss,
                         discriminator.accuracy],
                        feed_dict)

                    line = ("%s: DIS step %d, loss %f" % (
                        datetime.datetime.now().isoformat(), step, current_loss))

                    loss_log.write(line + "\n")
                    loss_log.flush()

                for g_epoch in range(FLAGS.g_epochs_num):
                    graphListIndex = getListOfGraphs()

                    features, support, yTrain = loadCandidateSamples(graphListIndex, featureList, supportList,
                                                                     y_train)
                    numberOfChosenGraph = (FLAGS.pools_size*FLAGS.number_of_vertices)/FLAGS.pools_size
                    feed_dict = construct_feed_dict(features[0], support, yTrain, numberOfChosenGraph, placeholders)

                    predicted = sess.run(generator.gan_score, feed_dict)
                    exp_rating = np.exp(np.array(predicted) * FLAGS.sampled_temperature)
                    prob = exp_rating / np.sum(exp_rating)

                    genIndex = np.random.choice(graphListIndex, size=FLAGS.gan_k, p=prob, replace=False)
                    features, support, yTrain = loadCandidateSamples(genIndex, featureList, supportList, y_train)
                    numberOfChosenGraph = (FLAGS.pools_size*FLAGS.number_of_vertices)/FLAGS.gan_k
                    feed_dict = construct_feed_dict(features[0], support, yTrain, numberOfChosenGraph, placeholders)

                    reward = sess.run(discriminator.reward, feed_dict)

                    print('reward')
                    print(reward)

                    feed_dict.update({generator.reward: reward})
                    feed_dict.update({generator.index: genIndex})
                    loss = sess.run(generator.gan_loss, feed_dict)

                    line = ("%s: GEN step %d, loss %f" % (
                        datetime.datetime.now().isoformat(), i+1, loss))

                    loss_log.write(line + "\n")
                    loss_log.flush()

main()